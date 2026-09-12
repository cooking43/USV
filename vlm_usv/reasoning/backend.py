"""Fast batched Qwen2-VL backend for the decision dump.

``src.vlm.vlm_inference.VLMInference`` answers one prompt per call and loads
the model in a way that is convenient for a demo and expensive for six
thousand decisions.  Measured on an RTX 3060 it costs 4.5 to 6.5 s per
decision, which would put the dump at eight to twenty hours and is an order of
magnitude away from the latency the system is supposed to have.

Four things account for most of it, and all four are fixable here.

*Slow image processor.*  The checkpoint ships a slow processor and the loader
does not ask for the fast one, so every frame is resized in Python.

*Unpinned visual resolution.*  Qwen2-VL maps an image to a token count that
grows with resolution, so a 1920x1080 frame becomes roughly 2,600 visual
tokens rather than the 256 the design assumes, and prefill grows with it.

*Dispatch hooks.*  ``device_map="auto"`` wraps every module in an accelerate
hook.  A 2B model in half precision fits on the card whole, so the hooks buy
nothing and cost a Python call per module per token.

*One decision at a time.*  Decoding is memory-bound: the weights are read once
per token whether one sequence is being decoded or thirty.  Episodes in the
replay set are independent and their decision instants coincide, so they batch
exactly.

Batching the language model is not the same as batching the vision tower, and
the two have to be separated, in both directions.  Too many images at once
overflows the card; too few leave it idle, because a tower call on two images
is far too small to saturate it and the calls run one after another.  The
chunk size is therefore measured rather than guessed, and the measurement is
counter-intuitive: encoding one image per call is fastest.  Under the dense
fallback the score matrix spans the whole concatenated chunk, so a chunk of
``c`` images at ``n`` patches each costs ``(cn)^2`` per call and ``B/c`` calls,
that is ``Bcn^2`` in total.  Everything above ``c = 1`` is off-diagonal work
that the mask then discards.  Batching the language model amortises weight
reads and is worth doing; batching the vision tower only manufactures work.  The tower does not composite the images: it
concatenates their patches into one sequence and masks attention across the
boundaries, so the result is per-image.  Under the SDPA fallback, however, the
masked score matrix is materialised at the full concatenated length, so memory
grows with the square of the whole batch rather than with the sum of the
squares of the images.  Sixteen frames at 256 merged tokens are 16,384 patches
before the merge, whose score matrix is 8.6 GiB in half precision across the
heads, which is what overflows a 12 GiB card.  Encoding the images a few at a
time and letting the language model keep the full batch removes the ceiling
without giving up the batching that matters.

The compact answer format contributes as well.  Five labelled lines cost about
thirty generated tokens; the same decision as one delimited line costs about
twelve, and decode time falls with it.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_VISUAL_TOKENS = 256
PATCH_AREA = 28 * 28          # pixels per visual token after the 2x2 merge


class QwenBatchRunner:
    """Loads Qwen2-VL once and answers a list of prompts in one forward pass."""

    def __init__(self, model_path: str, device: str = "cuda",
                 dtype: str = "float16", max_new_tokens: int = 16,
                 stop_at_newline: bool = True,
                 visual_tokens: int = DEFAULT_VISUAL_TOKENS,
                 batch_size: int = 8, vision_chunk: int = 1,
                 prefix_cache: bool = False, quantization: str = "none",
                 static_cache: bool = False, fast_decode: bool = True,
                 cuda_graph: bool = False,
                 # Off, and left in place with the check that retires it.  The
                 # capture succeeds and reproduces the plain prefill on the
                 # scene it was captured from, then diverges on the next one,
                 # and the cause was not found.  The comparison against the
                 # plain prefill on the first scenes is what keeps it out of
                 # the results rather than a claim that it is correct.
                 graph_prefill: bool = False,
                 prefill_pad: int = 704, max_context: int = 1024,
                 verbose: bool = True):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                              "expandable_segments:True")
        import torch
        from transformers import (AutoProcessor,
                                  Qwen2VLForConditionalGeneration)

        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.stop_at_newline = stop_at_newline
        self.batch_size = batch_size
        self.vision_chunk = vision_chunk
        self.use_prefix_cache = prefix_cache
        self._prefix = None            # (text, n_tokens, DynamicCache)
        self._prefix_verified = None   # None until the first comparison
        self.fast_decode = fast_decode
        self._decode_verified = None
        self.use_cuda_graph = cuda_graph
        self.max_context = max_context
        self._graph = None
        self._graph_failed = None
        self._vision_graph = None
        self._vision_graph_failed = True   # not wired in, see _graph_vision
        self.graph_prefill = graph_prefill
        self.prefill_pad = prefill_pad
        self._pf_graph = None
        self._pf_failed = None
        self._pf_checked = 0
        self.visual_tokens = visual_tokens
        self.stats = {"calls": 0, "sequences": 0, "prefill_tokens": 0,
                      "generated_tokens": 0, "seconds": 0.0,
                      "render_s": 0.0, "preprocess_s": 0.0, "vision_s": 0.0,
                      "generate_s": 0.0, "decode_s": 0.0,
                      "prefill_s": 0.0, "decode_steps": 0}

        t0 = time.time()
        budget = visual_tokens * PATCH_AREA
        # min and max are set to the same budget so the token count is fixed
        # rather than merely bounded, which keeps the accounting exact
        self.processor = AutoProcessor.from_pretrained(
            model_path, use_fast=True,
            min_pixels=budget, max_pixels=budget)
        # decoder-only batching requires the shorter sequences to be padded on
        # the left, or generation continues from padding
        self.processor.tokenizer.padding_side = "left"

        torch_dtype = getattr(torch, dtype)
        self.quantization = quantization
        self.static_cache = static_cache
        load_kw = dict(torch_dtype=torch_dtype, attn_implementation="sdpa")
        if quantization in ("int8", "int4"):
            # Decoding reads every weight once per token, so it is bound by
            # memory bandwidth rather than arithmetic, and narrower weights buy
            # time in direct proportion.  The vision tower is left in half
            # precision: it is compute-bound and quantising it would cost
            # accuracy for no speed.
            from transformers import BitsAndBytesConfig      # noqa: PLC0415
            # The vision tower is left in half precision: it is compute-bound,
            # so narrower weights buy it nothing, and an earlier attempt to
            # quantise it left its state uninitialised.
            skip = ["visual", "lm_head"]
            if quantization == "int8":
                bnb = BitsAndBytesConfig(load_in_8bit=True,
                                         llm_int8_skip_modules=skip,
                                         llm_int8_threshold=6.0)
            else:
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    llm_int8_skip_modules=skip)
            load_kw["quantization_config"] = bnb
            # accelerate places and initialises the quantised layers; moving
            # the module afterwards leaves their state uninitialised, which
            # surfaces much later as "quantization state not initialized"
            load_kw["device_map"] = "auto"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, **load_kw)
        if quantization == "none":
            self.model.to(device)
        self.model.eval()
        # the checkpoint ships sampling parameters that warn on every call
        # under greedy decoding
        for k in ("temperature", "top_p", "top_k"):
            if hasattr(self.model.generation_config, k):
                setattr(self.model.generation_config, k, None)
        self.model.generation_config.do_sample = False
        self._chunk_vision_tower()
        self._stop = self._newline_stopper()
        if verbose:
            print(f"    backend ready in {time.time() - t0:.1f}s  "
                  f"[{dtype}/{quantization}, sdpa, fast processor, "
                  f"{visual_tokens} visual tokens, batch {batch_size}, "
                  f"vision chunk {vision_chunk}]", flush=True)

    def _newline_stopper(self):
        """Stop as soon as every sequence has finished its line.

        The answer is one delimited line, so everything generated after the
        first newline is discarded by the parser and paid for by the decoder.
        Decoding is sequential in the number of steps regardless of batch, so
        this is the one cost that batching cannot amortise, and cutting the
        budget from twenty-four steps to the length of an actual answer is
        worth more than any further increase in batch size.
        """
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        tok = self.processor.tokenizer

        def resolve():
            ids = set()
            for piece in ("\n", "Ċ", ".\n"):
                try:
                    enc = tok.encode(piece, add_special_tokens=False)
                except Exception:                          # noqa: BLE001
                    continue
                if len(enc) == 1:
                    ids.add(enc[0])
            for t in ("<|im_end|>", "<|endoftext|>"):
                i = tok.convert_tokens_to_ids(t)
                if isinstance(i, int) and i >= 0:
                    ids.add(i)
            return ids

        newline_ids = resolve()
        self.stop_token_ids = sorted(newline_ids)
        # a configuration that reasons before it commits needs the newline to
        # be an ordinary token; only the end-of-turn markers end the answer
        self.eos_token_ids = sorted(
            i for i in newline_ids
            if i in {tok.convert_tokens_to_ids("<|im_end|>"),
                     tok.convert_tokens_to_ids("<|endoftext|>")})

        class LineDone(StoppingCriteria):
            def __init__(self, ids, prompt_len):
                self.ids = torch.tensor(sorted(ids))
                self.prompt_len = prompt_len

            def __call__(self, input_ids, scores, **kw):
                gen = input_ids[:, self.prompt_len:]
                if gen.numel() == 0:
                    return False
                hit = (gen.unsqueeze(-1) == self.ids.to(gen.device)).any(-1)
                return bool(hit.any(dim=1).all())

        self._LineDone = LineDone
        self._StoppingCriteriaList = StoppingCriteriaList
        return True

    def release(self) -> None:
        """Drop the captured graphs before the runner goes away.

        A captured graph owns a private memory pool.  Letting the runner be
        collected with a graph still referenced leaves that pool alive, and the
        next capture in the same process fails with a complaint about a
        previous error, which is what killed the quantised configurations.
        """
        self._graph = None
        self._vision_graph = None
        self._pf_graph = None
        self._graph_failed = None
        self._pf_failed = None
        self._pf_checked = 0
        try:
            self.torch.cuda.synchronize()
            self.torch.cuda.empty_cache()
        except Exception:                                   # noqa: BLE001
            pass

    def reset_stats(self) -> None:
        """Zero the counters without rebinding the dictionary."""
        for k in self.stats:
            self.stats[k] = 0.0 if isinstance(self.stats[k], float) else 0

    # ------------------------------------------------------------ vision graph
    def warmup_graphs(self, sample_image) -> bool:
        """No-op: the vision tower cannot be captured, and trying poisons the
        context.

        Two attempts settled this.  The tower builds its cross-image mask by
        slicing with values taken from ``cu_seqlens``, which forces a
        device-to-host synchronisation, and synchronisation is illegal inside a
        capture.  The capture therefore fails wherever it is placed, inside the
        forward or in a warm-up of its own, and a failed capture leaves the
        context in a state where the decode capture fails too.  Since the
        decode loop was 1200 ms against the tower's 130, giving up the tower to
        keep the decode graph is the right trade.
        """
        return False

    def _replay_vision(self, pixel_values):
        g = self._vision_graph
        g["px"].copy_(pixel_values)
        g["graph"].replay()
        return g["out"]

    def _graph_vision(self, pixel_values, grid_thw):
        """Capture the vision tower. Not wired in; see the note below.

        The argument for graphing the tower is the same as for the decode
        step, and the shape is fixed once the token budget is pinned.  What
        does not work is capturing it from inside ``visual.forward``: that call
        happens in the middle of the prefill's own forward pass, and starting a
        capture there is a nested capture, which fails and leaves the context
        poisoned so that the decode capture fails afterwards too.  Capturing it
        would have to happen in a separate warm-up phase before any forward is
        in flight.  The tower is 130 ms of the budget against the decode
        loop's 1200 ms, so it was not worth the surgery.
        """
        torch = self.torch
        key = (tuple(pixel_values.shape), tuple(grid_thw.shape))
        cached = self._vision_graph
        if cached is None or cached["key"] != key:
            original = self._visual_forward
            static_px = pixel_values.clone()
            static_g = grid_thw.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    with torch.no_grad():
                        original(static_px, static_g)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                with torch.no_grad():
                    out = original(static_px, static_g)
            cached = {"key": key, "graph": graph, "px": static_px,
                      "grid": static_g, "out": out}
            self._vision_graph = cached
        cached["px"].copy_(pixel_values)
        cached["graph"].replay()
        return cached["out"]

    # ------------------------------------------------------------ cuda graph
    def _build_graph(self, prompt_len: int):
        """Capture one decode step as a CUDA graph.

        The probe settles what the cost is: a single small operation takes
        165 microseconds to dispatch on this platform, and a decode step
        issues on the order of seven hundred of them, which is the 127 ms
        measured and the reason the GPU sits at thirteen percent utilisation
        while it runs.  The work is not the problem; the number of times the
        driver is asked to do work is.

        A captured graph replays the whole step as one submission.  Capture
        needs shapes and addresses to be fixed, so the key-value cache is
        preallocated and written in place at an index the graph reads from a
        tensor rather than from Python.
        """
        torch = self.torch
        from transformers import StaticCache

        dev = self.device
        cache = StaticCache(config=self.model.config, max_batch_size=1,
                            max_cache_len=self.max_context, device=dev,
                            dtype=self.model.dtype)
        g_ids = torch.zeros(1, 1, dtype=torch.long, device=dev)
        g_pos = torch.zeros(3, 1, 1, dtype=torch.long, device=dev)
        g_cpos = torch.zeros(1, dtype=torch.long, device=dev)
        g_mask = torch.zeros(1, self.max_context, dtype=torch.long, device=dev)

        def one_step():
            return self.model(input_ids=g_ids, position_ids=g_pos,
                              attention_mask=g_mask,
                              past_key_values=cache, cache_position=g_cpos,
                              use_cache=True)

        # warm up on a side stream, which capture requires
        g_cpos.fill_(prompt_len)
        g_mask[:, : prompt_len + 1] = 1
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                with torch.no_grad():
                    one_step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with torch.no_grad():
                out = one_step()
            logits = out.logits
        self._graph = {"graph": graph, "cache": cache, "ids": g_ids,
                       "pos": g_pos, "cpos": g_cpos, "mask": g_mask,
                       "logits": logits, "prompt_len": prompt_len}
        return self._graph

    def _pad_inputs(self, inputs, pad_len: int):
        """Right-pad the prompt to a fixed length so the prefill has one shape.

        The scene description varies in length from frame to frame, which is
        the only thing that stops the prefill being captured.  Padding on the
        right rather than the left is what makes it correct as well as fixed:
        attention is causal, so a real token cannot see a position that comes
        after it, and the padding is inert without the two-dimensional mask
        having to be honoured at all.  Left padding was tried first and failed
        in a way worth recording, since it failed quietly -- a prompt padded by
        two tokens agreed with the plain prefill and one padded by a hundred
        and thirty-nine did not, which is the signature of real tokens reading
        the padding rather than of a broken capture.
        """
        torch = self.torch
        ids = inputs["input_ids"]
        n = int(ids.shape[1])
        if n > pad_len:
            return None
        pad_id = self.processor.tokenizer.pad_token_id or 0
        right = torch.full((1, pad_len - n), pad_id, dtype=ids.dtype,
                           device=ids.device)
        out = dict(inputs)
        out["input_ids"] = torch.cat([ids, right], dim=1)
        mask = inputs.get("attention_mask")
        if mask is None:
            mask = torch.ones_like(ids)
        out["attention_mask"] = torch.cat([mask, torch.zeros_like(right)],
                                          dim=1)
        out["_n_real"] = n
        return out

    def _prefill_embeds(self, padded):
        """Everything in the prefill that cannot be captured, done outside it.

        Two steps in a multimodal prefill reach back to the host: the image
        embeddings are scattered into the token embeddings at the positions
        where ``input_ids`` equals the image token, and the multimodal rotary
        positions are derived by walking the grid on the CPU.  Both read a
        tensor's values to decide what to do, which is a device-host
        synchronisation and is illegal inside a capture; an earlier attempt to
        capture the prefill whole failed on exactly this and left the capture
        open, which poisoned the context and destroyed the working decode
        graph with it.

        Doing them here leaves a pure tensor computation over fixed shapes,
        which captures cleanly.
        """
        torch = self.torch
        model = self.model
        ids = padded["input_ids"]
        mask = padded["attention_mask"]
        # the mask handed to get_rope_index must be all ones over the padded
        # length, so the real tokens keep the positions they would have had
        # and the padding simply continues the sequence

        with torch.no_grad():
            embeds = model.get_input_embeddings()(ids)
            pv = padded.get("pixel_values")
            if pv is not None:
                grid = padded["image_grid_thw"]
                img = model.visual(pv, grid_thw=grid).to(embeds.dtype)
                sel = (ids == model.config.image_token_id)
                embeds = embeds.masked_scatter(
                    sel.unsqueeze(-1).expand_as(embeds), img)
            pos, delta = model.get_rope_index(
                ids, padded.get("image_grid_thw"), None, mask)
        return embeds, pos, delta

    def _prefill_mask(self, n):
        """All ones over the padded length.

        The padding sits after every real token and causality already hides
        it, so there is nothing for this mask to exclude; making it uniform
        also makes it identical for every scene, which is one fewer thing the
        capture can be sensitive to.
        """
        torch = self.torch
        full = torch.zeros(1, self.max_context, dtype=torch.long,
                           device=self.device)
        full[:, :n] = 1
        return full

    def _full_mask(self, mask, n):
        """A mask as wide as the cache, which is what StaticCache indexes.

        Handing the decoder a mask only as wide as the prompt while the cache
        is longer makes the causal-mask builder read past the end of it, which
        surfaces as a device-side assert with no useful stack.
        """
        torch = self.torch
        full = torch.zeros(1, self.max_context, dtype=mask.dtype,
                           device=mask.device)
        full[:, :n] = mask
        return full

    def _build_prefill_graph(self, embeds, pos, mask, cache):
        """Capture the language-model half of the prefill.

        Capture is begun and ended explicitly rather than through the context
        manager, so that a failure ends the capture on the way out.  A capture
        left open makes every later allocation on the device raise, including
        the ones belonging to paths that were working.
        """
        torch = self.torch
        n = embeds.shape[1]
        st = {"embeds": embeds.clone(), "pos": pos.clone(),
              "mask": self._prefill_mask(n)}
        cpos = torch.arange(n, device=self.device)

        def once():
            cache.reset()
            with torch.no_grad():
                return self.model.model(inputs_embeds=st["embeds"],
                                        position_ids=st["pos"],
                                        attention_mask=st["mask"],
                                        past_key_values=cache,
                                        cache_position=cpos, use_cache=True)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                once()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        began = False
        cap = torch.cuda.Stream()
        cap.wait_stream(torch.cuda.current_stream())
        try:
            # capture is only legal off the default stream, and capture_begin
            # does not move there on its own the way the context manager does
            with torch.cuda.stream(cap):
                cache.reset()
                graph.capture_begin()
                began = True
                with torch.no_grad():
                    out = self.model.model(inputs_embeds=st["embeds"],
                                           position_ids=st["pos"],
                                           attention_mask=st["mask"],
                                           past_key_values=cache,
                                           cache_position=cpos, use_cache=True)
                hidden = (out.last_hidden_state
                          if hasattr(out, "last_hidden_state") else out[0])
                graph.capture_end()
                began = False
        except Exception:
            if began:
                try:
                    graph.capture_end()
                except Exception:                              # noqa: BLE001
                    pass
            raise
        torch.cuda.current_stream().wait_stream(cap)
        torch.cuda.synchronize()
        self._pf_graph = {"graph": graph, "static": st, "hidden": hidden,
                          "pad_len": n}
        return self._pf_graph

    def _stops(self):
        """Which tokens end an answer, given how this run is configured."""
        return (self.stop_token_ids if self.stop_at_newline
                else (self.eos_token_ids or self.stop_token_ids))

    def _greedy_graphed(self, inputs, stop_ids, max_new: int):
        """Prefill normally, then replay the captured step for each token."""
        torch = self.torch
        ids = inputs["input_ids"]
        batch, prompt_len = ids.shape
        if batch != 1:
            raise RuntimeError("the captured step is for one frame at a time")

        # The captured step is independent of the prompt length: that length
        # reaches the graph only through the values of the cache-position and
        # mask tensors, whose shapes are fixed.  Rebuilding per length cost
        # about 200 ms a frame for nothing.
        g = self._graph
        if g is None:
            g = self._build_graph(prompt_len)
        cache = g["cache"]
        cache.reset()

        # prefill writes into the same preallocated cache
        padded = (self._pad_inputs(inputs, self.prefill_pad)
                  if (self.graph_prefill and self._pf_failed is not True)
                  else None)
        t_pf = time.time()
        used_pf = False
        if padded is not None:
            try:
                tensors = {k: v for k, v in padded.items()
                           if not k.startswith("_")}
                embeds, pos, delta_pf = self._prefill_embeds(tensors)
                if self._pf_graph is None:
                    self._build_prefill_graph(embeds, pos,
                                              tensors["attention_mask"], cache)
                pg = self._pf_graph
                pg["static"]["embeds"].copy_(embeds)
                pg["static"]["pos"].copy_(pos)
                pg["static"]["mask"].copy_(self._prefill_mask(pg["pad_len"]))
                cache.reset()
                pg["graph"].replay()
                n_real = int(padded["_n_real"])
                out_logits = self.model.lm_head(
                    pg["hidden"][:, n_real - 1:n_real, :])
                prompt_len = n_real
                # the offset the decode loop adds is taken from the real
                # prefix, not from the padded length; the padding continues
                # the position sequence and would otherwise be counted
                nxt = int(pos[:, :, :n_real].max()) + 1
                delta_pf = torch.tensor([[nxt - n_real]], device=self.device)
                self.model.rope_deltas = delta_pf
                self._pf_failed = False
                used_pf = True
            except Exception as exc:                           # noqa: BLE001
                self._pf_failed = True
                self._pf_graph = None
                print(f"    [note] prefill graph unavailable: "
                      f"{type(exc).__name__}: {exc}", flush=True)
        def plain_prefill():
            """The unpadded prefill, which is the reference for the graph."""
            n = int(inputs["input_ids"].shape[1])
            g["mask"].zero_()
            g["mask"][:, :n] = 1
            with torch.no_grad():
                o = self.model(**{k: v for k, v in inputs.items()
                                  if k != "attention_mask"},
                               attention_mask=g["mask"][:, :n],
                               past_key_values=cache,
                               cache_position=torch.arange(n,
                                                           device=self.device),
                               use_cache=True)
            return o.logits, n

        if used_pf and self._pf_checked < 3:
            # The graph is captured once and then replayed with new values
            # copied into its inputs.  A capture that reads a stale address,
            # or a position index that only happens to be right for the length
            # it was captured at, is silent: the answer is merely different.
            # The first few scenes are therefore run both ways and compared on
            # the token the prefill actually decides, and a mismatch retires
            # the graph instead of being carried into the results.
            self._pf_checked += 1
            want = int(out_logits[:, -1, :].argmax(-1))
            ref_logits, ref_n = plain_prefill()
            got = int(ref_logits[:, -1, :].argmax(-1))
            if want != got:
                print(f"    [note] prefill graph disagrees on a new scene "
                      f"(graph {want} vs plain {got}, "
                      f"len {ref_n} padded to {prompt_len}); "
                      f"reverting to the plain prefill", flush=True)
                self._pf_failed = True
                self._pf_graph = None
                used_pf = False
                out_logits, prompt_len = ref_logits, ref_n
            else:
                if self._pf_checked == 1:
                    print("    [note] prefill graph matches the plain prefill",
                          flush=True)
                # the reference run overwrote the cache, so restore the padded
                # one the decode loop is about to continue from
                pg = self._pf_graph
                cache.reset()
                pg["graph"].replay()
                prompt_len = int(padded["_n_real"])
                self.model.rope_deltas = delta_pf

        if not used_pf:
            out_logits, prompt_len = plain_prefill()
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.stats["prefill_s"] += time.time() - t_pf

        g["mask"].zero_()
        # prompt_len is the real token count in both paths now, and the cache
        # slots the padding wrote into are simply never unmasked
        g["mask"][:, :prompt_len] = 1
        deltas = getattr(self.model, "rope_deltas", None)
        delta = int(deltas.reshape(-1)[0]) if deltas is not None else 0

        tok = out_logits[:, -1, :].argmax(-1, keepdim=True)
        produced = [tok.clone()]
        stop = torch.tensor(sorted(stop_ids), device=self.device)
        done = (tok == stop).any(-1)

        t_dec = time.time()
        for step in range(1, max_new):
            if bool(done.all()):
                break
            pos_val = prompt_len + step - 1 + delta
            g["ids"].copy_(tok)
            g["pos"].fill_(pos_val)
            g["cpos"].fill_(prompt_len + step - 1)
            g["mask"][:, prompt_len + step - 1] = 1
            g["graph"].replay()
            tok = g["logits"][:, -1, :].argmax(-1, keepdim=True)
            produced.append(tok.clone())
            done = done | (tok == stop).any(-1)
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.stats["decode_s"] += time.time() - t_dec
        self.stats["decode_steps"] += len(produced)
        return torch.cat([ids, torch.cat(produced, dim=1)], dim=1)

    # ------------------------------------------------------------ decode loop
    def _greedy(self, inputs, stop_ids, max_new: int):
        """Greedy decode without the generation machinery.

        Measured at batch one, a decode step costs about 127 ms for a 2B model
        in half precision on this card, against roughly 12 ms for reading the
        weights once, which is the memory-bandwidth floor.  The gap is not
        arithmetic and it is not bandwidth; it is the work done between steps.
        For a vision-language model the largest part of that is the multimodal
        rotary index, which the generation path rebuilds on every step from the
        whole token sequence and the image grid.

        The positions are an arithmetic progression once the prefill has fixed
        their offset, so they are computed once and incremented.  Everything
        else the loop needs is the key-value cache the previous step returned.
        """
        torch = self.torch
        model = self.model
        ids = inputs["input_ids"]
        attn = inputs.get("attention_mask")
        batch, prompt_len = ids.shape

        t_pf = time.time()
        out = model(**inputs, use_cache=True)
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.stats["prefill_s"] += time.time() - t_pf
        past = out.past_key_values
        deltas = getattr(model, "rope_deltas", None)
        if deltas is None:
            deltas = torch.zeros(batch, 1, device=ids.device, dtype=torch.long)
        deltas = deltas.reshape(batch, 1).to(ids.device)

        tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        produced = [tok]
        stop = torch.tensor(sorted(stop_ids), device=ids.device)
        done = (tok == stop).any(-1)

        t_dec = time.time()
        for step in range(1, max_new):
            if bool(done.all()):
                break
            if attn is not None:
                attn = torch.cat([attn, torch.ones_like(tok)], dim=1)
            abs_pos = prompt_len + step - 1
            pos = (torch.full((batch, 1), abs_pos, device=ids.device,
                              dtype=torch.long) + deltas)
            pos = pos.unsqueeze(0).expand(3, batch, 1)
            out = model(input_ids=tok, attention_mask=attn, position_ids=pos,
                        past_key_values=past, use_cache=True)
            past = out.past_key_values
            tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
            produced.append(tok)
            done = done | (tok == stop).any(-1)

        if self.device == "cuda":
            torch.cuda.synchronize()
        self.stats["decode_s"] += time.time() - t_dec
        self.stats["decode_steps"] += len(produced)
        gen = torch.cat(produced, dim=1)
        return torch.cat([ids, gen], dim=1)

    # ------------------------------------------------------------ generation
    def _run_generate(self, inputs, stopping, system, shots):
        """Generate, reusing the constant prefix's keys and values if it is safe.

        Reuse is verified rather than assumed.  Qwen2-VL derives its
        multimodal rotary positions from the whole ``input_ids``, and a
        pre-filled cache sends that computation down a different branch, so a
        mistake would not raise: it would shift the positions and quietly
        change the decision.  The first call therefore runs both ways and
        compares; reuse stays on only if the token sequences match exactly,
        and otherwise it is disabled for the rest of the run and reported.
        """
        gen = dict(max_new_tokens=self.max_new_tokens, do_sample=False,
                   stopping_criteria=stopping,
                   pad_token_id=self.processor.tokenizer.pad_token_id)

        if self.use_cuda_graph and self._graph_failed is None:
            try:
                ref = self.model.generate(**inputs, **gen)
                mine = self._greedy_graphed(inputs, self._stops(),
                                            self.max_new_tokens)
                n = min(ref.shape[1], mine.shape[1])
                if bool((ref[:, :n] == mine[:, :n]).all()):
                    self._graph_failed = False
                    print("    [note] cuda graph matches the generation path",
                          flush=True)
                    self.use_cuda_graph_ready = True
                    return mine
                self._graph_failed = True
                print("    [note] cuda graph output differs; disabled",
                      flush=True)
                return ref
            except Exception as exc:                       # noqa: BLE001
                self._graph_failed = True
                print(f"    [note] cuda graph unavailable: "
                      f"{type(exc).__name__}: {exc}", flush=True)
        elif self.use_cuda_graph and self._graph_failed is False:
            return self._greedy_graphed(inputs, self._stops(),
                                        self.max_new_tokens)

        if self.fast_decode:
            if self._decode_verified is None:
                # verified once against the generation path, for the same
                # reason the prefix cache is: a positional mistake would not
                # raise, it would quietly change the decision
                ref = self.model.generate(**inputs, **gen)
                mine = self._greedy(inputs, self._stops(),
                                    self.max_new_tokens)
                n = min(ref.shape[1], mine.shape[1])
                same = bool((ref[:, :n] == mine[:, :n]).all())
                self._decode_verified = same
                print(f"    [note] fast decode "
                      f"{'matches' if same else 'differs from'} the "
                      f"generation path", flush=True)
                if not same:
                    self.fast_decode = False
                return ref
            if self._decode_verified:
                return self._greedy(inputs, self._stops(),
                                    self.max_new_tokens)

        if not self.use_prefix_cache:
            return self.model.generate(**inputs, **gen)

        plain = self.model.generate(**inputs, **gen)
        if self._prefix_verified is True:
            return self._generate_with_prefix(inputs, gen, system, shots, plain)
        if self._prefix_verified is False:
            return plain

        try:
            cached = self._generate_with_prefix(inputs, gen, system, shots, None)
            same = (cached.shape == plain.shape and bool((cached == plain).all()))
        except Exception as exc:                            # noqa: BLE001
            same, cached = False, None
            print(f"    [note] prefix reuse raised {type(exc).__name__}: {exc}",
                  flush=True)
        self._prefix_verified = same
        if same:
            print("    [note] prefix reuse verified against the plain path",
                  flush=True)
            return cached
        print("    [note] prefix reuse changes the output; disabled", flush=True)
        self.use_prefix_cache = False
        return plain

    def _generate_with_prefix(self, inputs, gen, system, shots, _plain):
        torch = self.torch
        _text, n_prefix, cache = self._build_prefix(system, shots)
        batch = int(inputs["input_ids"].shape[0])
        past = self._expand_prefix(cache, batch)
        kw = dict(inputs)
        kw["past_key_values"] = past
        kw["cache_position"] = torch.arange(
            n_prefix, int(inputs["input_ids"].shape[1]), device=self.device)
        self.model.rope_deltas = None
        return self.model.generate(**kw, **gen)

    # ------------------------------------------------------------ prefix cache
    def _prefix_text(self, system: str, shots) -> str:
        """The part of the rendered prompt that never varies.

        The instruction block and the worked examples are identical on every
        frame and sit before the image, so their attention keys and values can
        be computed once and reused.  Only the part that describes the scene
        has to be prefilled per decision.  The split is exactly the accounting
        the efficiency table uses: the constant block is a fixed cost of the
        deployment, the scene description is the per-frame cost.
        """
        rendered = self._render(system, shots, "\u0000SCENE\u0000", True)
        head, _sep, _tail = rendered.partition("\u0000SCENE\u0000")
        # cut back to the last turn boundary so the cached span ends cleanly
        marker = "<|im_start|>user"
        idx = head.rfind(marker)
        return head[:idx] if idx > 0 else head

    def _build_prefix(self, system: str, shots):
        torch = self.torch
        text = self._prefix_text(system, shots)
        if self._prefix is not None and self._prefix[0] == text:
            return self._prefix
        ids = self.processor.tokenizer(text, return_tensors="pt",
                                       add_special_tokens=False)
        ids = {k: v.to(self.device) for k, v in ids.items()}
        with torch.no_grad():
            out = self.model(**ids, use_cache=True)
        self._prefix = (text, int(ids["input_ids"].shape[1]),
                        out.past_key_values)
        return self._prefix

    def _expand_prefix(self, cache, batch: int):
        """A per-call copy of the cached keys and values, widened to the batch."""
        torch = self.torch
        from transformers import DynamicCache
        new = DynamicCache()
        for layer, (k, v) in enumerate(zip(cache.key_cache, cache.value_cache)):
            new.update(k.expand(batch, -1, -1, -1).contiguous(),
                       v.expand(batch, -1, -1, -1).contiguous(), layer)
        return new

    def set_visual_tokens(self, n: int) -> None:
        """Repin the visual token budget without reloading the weights.

        The budget lives in the image processor, not in the model, so a sweep
        over it costs nothing but a re-resize of the images.  Reloading for
        each value would spend three minutes of disk on a change that touches
        two integers.
        """
        self.visual_tokens = n
        budget = n * PATCH_AREA
        ip = getattr(self.processor, "image_processor", None)
        if ip is not None:
            ip.max_pixels = budget
            ip.min_pixels = budget
            size = getattr(ip, "size", None)
            if isinstance(size, dict):
                size["longest_edge"] = budget
                size["shortest_edge"] = budget

    # ------------------------------------------------------------ images
    def prepare_image(self, image):
        """Resize once to the pinned budget, so the processor does no work.

        The processor resizes every frame to the pixel budget on every call.
        At a 1920x1080 input that is a CPU-side resample costing tens of
        milliseconds per decision, and it repeats identically for a frame that
        several decisions share.  Doing it here lets the caller cache the
        result; the processor then receives an image already at the target
        size and its own resize is a no-op.
        """
        if image is None:
            return None
        budget = self.visual_tokens * PATCH_AREA
        w, h = image.size
        if abs(w * h - budget) / budget < 0.02 and w % 28 == 0 and h % 28 == 0:
            return image
        import math
        scale = math.sqrt(budget / float(w * h))
        nw = max(28, int(round(w * scale / 28)) * 28)
        nh = max(28, int(round(h * scale / 28)) * 28)
        try:
            from PIL import Image as _I
            resample = _I.Resampling.BICUBIC
        except Exception:                                   # noqa: BLE001
            resample = 3
        return image.resize((nw, nh), resample)

    # ------------------------------------------------------------ vision tower
    def _chunk_vision_tower(self) -> None:
        """Encode the batch's images a few at a time, and time the tower.

        The tower's own forward is replaced rather than the model's, so the
        language model still receives ``input_ids`` and ``pixel_values`` and
        computes its multimodal rotary positions exactly as before.  Only the
        order in which the patches reach the encoder changes, and since
        attention never crosses an image boundary the embeddings are identical
        to those the unchunked call would produce.
        """
        torch = self.torch
        visual = getattr(self.model, "visual", None)
        if visual is None:
            return
        original = getattr(self, "_visual_forward", None) or visual.forward
        self._visual_forward = original
        runner = self
        stats = self.stats

        def forward(pixel_values, grid_thw, *a, **kw):
            chunk = runner.vision_chunk
            t0 = time.time()
            if chunk <= 0 or grid_thw is None or len(grid_thw) <= chunk:
                out = original(pixel_values, grid_thw, *a, **kw)
                stats["vision_s"] += time.time() - t0
                return out
            counts = grid_thw.prod(-1).tolist()
            parts, lo = [], 0
            for i in range(0, len(counts), chunk):
                n = int(sum(counts[i:i + chunk]))
                parts.append(original(pixel_values[lo:lo + n],
                                      grid_thw[i:i + chunk], *a, **kw))
                lo += n
            stats["vision_s"] += time.time() - t0
            return torch.cat(parts, dim=0)

        visual.forward = forward

    # ---------------------------------------------------------------- prompts
    def _render(self, system: str, shots: Sequence[Tuple[str, str]],
                prompt: str, with_image: bool) -> str:
        messages = [{"role": "system", "content": system}]
        for u, a in shots:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        content = ([{"type": "image"}, {"type": "text", "text": prompt}]
                   if with_image else prompt)
        messages.append({"role": "user", "content": content})
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    # ---------------------------------------------------------------- generate
    def generate(self, prompts: Sequence[str], images: Sequence,
                 system: str, shots: Sequence[Tuple[str, str]] = ()) -> List[str]:
        """One reply per prompt. ``images[i]`` may be ``None``.

        The chunk size halves and retries on an out-of-memory error rather
        than aborting.  The limit is the vision tower, not the language model:
        it concatenates the batch into one sequence of patches and attends
        over all of them, so its activation grows with the square of the batch
        and a size that fits on one machine overflows on another.  A dump that
        died two hours in because the batch was one too large would be worse
        than a dump that ran slightly slower.
        """
        out: List[str] = []
        lo = 0
        chunk = self.batch_size
        while lo < len(prompts):
            hi = min(lo + chunk, len(prompts))
            try:
                out.extend(self._generate_chunk(prompts[lo:hi], images[lo:hi],
                                                system, shots))
            except Exception as exc:                       # noqa: BLE001
                if "out of memory" not in str(exc).lower() or chunk == 1:
                    raise
                self.torch.cuda.empty_cache()
                if self.vision_chunk > 1:
                    # the tower is the usual culprit, so shrink it before
                    # giving up throughput on the language model
                    self.vision_chunk = max(1, self.vision_chunk // 2)
                    print(f"    [note] out of memory at batch {hi - lo}, "
                          f"vision chunk -> {self.vision_chunk}", flush=True)
                    self._chunk_vision_tower()
                else:
                    chunk = max(1, chunk // 2)
                    print(f"    [note] out of memory at batch {hi - lo}, "
                          f"batch -> {chunk}", flush=True)
                continue
            lo = hi
        self.batch_size = min(self.batch_size, chunk)
        return out

    def _generate_chunk(self, prompts, images, system, shots) -> List[str]:
        torch = self.torch
        t_render = time.time()
        with_image = [im is not None for im in images]
        texts = [self._render(system, shots, p, wi)
                 for p, wi in zip(prompts, with_image)]
        imgs = [im for im in images if im is not None]
        self.stats["render_s"] += time.time() - t_render

        t_pre = time.time()
        inputs = self.processor(text=texts, images=imgs or None,
                                padding=True, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.stats["preprocess_s"] += time.time() - t_pre

        n_prompt = int(inputs["input_ids"].shape[1])
        stopping = self._StoppingCriteriaList(
            [self._LineDone(self._stops(), n_prompt)])
        t0 = time.time()
        with torch.no_grad():
            ids = self._run_generate(inputs, stopping, system, shots)
        if self.device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        self.stats["generate_s"] += dt

        t_dec = time.time()
        n_in = int(inputs["input_ids"].shape[1])
        trimmed = ids[:, n_in:]
        replies = self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        self.stats["decode_s"] += time.time() - t_dec

        self.stats["calls"] += 1
        self.stats["sequences"] += len(prompts)
        self.stats["prefill_tokens"] += n_in * len(prompts)
        self.stats["generated_tokens"] += int((trimmed != self.processor
                                               .tokenizer.pad_token_id).sum())
        self.stats["seconds"] += dt
        return replies

    # ---------------------------------------------------------------- report
    def timing(self) -> Dict:
        s = self.stats
        n = max(s["sequences"], 1)
        return {
            "sequences": s["sequences"],
            "batched_calls": s["calls"],
            "mean_batch": round(n / max(s["calls"], 1), 1),
            "seconds_total": round(s["seconds"], 1),
            "seconds_per_decision": round(s["seconds"] / n, 3),
            "prefill_tokens_per_decision": round(s["prefill_tokens"] / n, 1),
            "generated_tokens_per_decision": round(s["generated_tokens"] / n, 1),
            "steps_per_decision": round(
                s["generated_tokens"] / max(s["sequences"], 1), 1),
            "phase_ms_per_decision": {
                "render": round(1000 * s["render_s"] / n, 1),
                "preprocess": round(1000 * s["preprocess_s"] / n, 1),
                "vision_tower": round(1000 * s["vision_s"] / n, 1),
                "generate_total": round(1000 * s["generate_s"] / n, 1),
                "generate_minus_vision": round(
                    1000 * (s["generate_s"] - s["vision_s"]) / n, 1),
                "text_decode": round(1000 * s["decode_s"] / n, 1),
                "prefill_fwd": round(1000 * s["prefill_s"] / n, 1),
                "decode_loop": round(1000 * s["decode_s"] / n, 1),
                "per_decode_step": round(
                    1000 * s["decode_s"] / max(s["decode_steps"], 1), 2),
            },
        }
