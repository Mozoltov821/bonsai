import jax.numpy as jnp
from typing import Tuple, Union, List


class InputSegment:

    def __init__(
            self,
            text: str = "",
            audio: jnp.ndarray = None,
            tokenized_text: jnp.ndarray = None,
            speech_zeroemb_idx: Union[int, List[int]] = 1024,
            text_zeroemb_idx: int = 152067,
            add_sosp_eosp=True,
    ) -> None:
        has_text = text is not None
        has_tokenized_text = tokenized_text is not None
        assert has_text or has_tokenized_text, "Text or tokenized text must be provided"

        self.audio = audio
        self.text = text
        self.tokenized_text = tokenized_text
        self.speech_zeroemb_idx = speech_zeroemb_idx
        self.text_zeroemb_idx = text_zeroemb_idx
        self.add_sosp_eosp = add_sosp_eosp

    @staticmethod
    def insert_between(tensor, i, value=-1):
        """Insert values between elements of tensor.

        Args:
            tensor: Input tensor of shape (1, n)
            i: Number of values to insert between elements
            value: Value to insert

        Returns:
            Tensor of shape (1, n + (n-1)*i + i)
        """
        n = tensor.shape[1]
        output_len = n + (n - 1) * i + i

        # Create output filled with value
        output = jnp.full((1, output_len), value, dtype=tensor.dtype)

        # Calculate positions for original elements
        positions = jnp.arange(0, n, dtype=jnp.int32) * (i + 1)

        # Place original elements at calculated positions
        output = output.at[0, positions].set(tensor[0, :])

        return output

    def to_input_id(
            self,
            tokenizer,
            group_size: int,
            audio_channels: int = 8,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.audio is None:
            if self.tokenized_text is None:
                tokenized_text = tokenizer(
                    self.text,
                    return_tensors="np",
                    truncation=True,
                    max_length=999999,
                    padding=False,
                    add_special_tokens=False,
                )["input_ids"]
                tokenized_text = jnp.array(tokenized_text, dtype=jnp.int32)
            else:
                tokenized_text = self.tokenized_text[jnp.newaxis, :]


            if group_size > 1:
                tokenized_text = self.insert_between(
                    tokenized_text, group_size - 1, value=-100
                )


            if isinstance(self.speech_zeroemb_idx, list):
                audio_part_input_id = jnp.zeros((audio_channels, tokenized_text.shape[1]), dtype=jnp.int32)
                for i, idx in enumerate(self.speech_zeroemb_idx):
                    audio_part_input_id = audio_part_input_id.at[i, :].set(idx)
            else:
                audio_part_input_id = jnp.full(
                    (audio_channels, tokenized_text.shape[1]), self.speech_zeroemb_idx, dtype=jnp.int32
                )


        else:
            sosp_token = (
                tokenizer.convert_tokens_to_ids("<|sosp|>")
                if self.add_sosp_eosp
                else None
            )
            eosp_token = (
                tokenizer.convert_tokens_to_ids("<|eosp|>")
                if self.add_sosp_eosp
                else None
            )
            audio_part = self.audio.reshape(-1, audio_channels).T  # [audio_channels, seqlen]

            assert (
                    audio_part.shape[1] % group_size == 0
            ), f"Audio shape {audio_part.shape} is not divisible by group_size {group_size}"


            text_len = audio_part.shape[1] // group_size
            empty_token = self.text_zeroemb_idx
            if empty_token is None:
                empty_token = tokenizer.eod
            tokenized_text = jnp.full((1, text_len), empty_token, dtype=jnp.int32)

            tokenized_text = (
                jnp.concatenate(
                    [
                        jnp.array([[sosp_token]], dtype=jnp.int32),
                        tokenized_text,
                        jnp.array([[eosp_token]], dtype=jnp.int32),
                    ],
                    axis=1,
                )
                if self.add_sosp_eosp
                else tokenized_text
            )
            tokenized_text = self.insert_between(
                tokenized_text, group_size - 1, value=-100
            )


            if self.add_sosp_eosp:
                if isinstance(self.speech_zeroemb_idx, list):
                    sosp_part = jnp.zeros((audio_channels, group_size), dtype=jnp.int32)
                    eosp_part = jnp.zeros((audio_channels, group_size), dtype=jnp.int32)
                    for i, idx in enumerate(self.speech_zeroemb_idx):
                        sosp_part = sosp_part.at[i, :].set(idx)
                        eosp_part = eosp_part.at[i, :].set(idx)
                    audio_part_input_id = jnp.concatenate([sosp_part, audio_part, eosp_part], axis=1)
                else:
                    audio_part_input_id = jnp.concatenate(
                        [
                            jnp.full((audio_channels, group_size), self.speech_zeroemb_idx, dtype=jnp.int32),
                            audio_part,
                            jnp.full((audio_channels, group_size), self.speech_zeroemb_idx, dtype=jnp.int32),
                        ],
                        axis=1,
                    )
            else:
                audio_part_input_id = audio_part



        input_ids = jnp.concatenate(
            [tokenized_text, audio_part_input_id], axis=0
        )  # [n_rvq + 1, seqlen]


        return input_ids


class StreamingInputSegment:
    def __init__(
            self,
            text: str = "",
            audio: jnp.ndarray = None,
            tokenized_text: jnp.ndarray = None,
            speech_zeroemb_idx: Union[int, List[int]] = 1024,
            text_zeroemb_idx: int = 152067,
            text_segment_size: int = 5,
            audio_segment_size: int = 5,
            tokenizer=None,
            group_size=None,
            audio_channels=None,
    ) -> None:
        has_text = text is not None
        has_tokenized_text = tokenized_text is not None
        assert has_text or has_tokenized_text, "Text or tokenized text must be provided"

        self.audio = audio
        self.text = text
        self.tokenized_text = tokenized_text
        self.speech_zeroemb_idx = speech_zeroemb_idx
        self.text_zeroemb_idx = text_zeroemb_idx
        self.text_segment_size = text_segment_size
        self.audio_segment_size = audio_segment_size
        self.tokenizer = tokenizer
        self.group_size = group_size
        self.audio_channels = audio_channels

    def to_input_id(
            self,
            tokenizer,
            group_size: int,
            audio_channels: int = 8,
    ):
        if self.tokenized_text is None:
            tokenized_text = tokenizer(
                self.text,
                return_tensors="np",
                truncation=True,
                max_length=999999,
                padding=False,
                add_special_tokens=False,
            )["input_ids"][0]  # [seqlen]
            tokenized_text = jnp.array(tokenized_text, dtype=jnp.int32)
        else:
            tokenized_text = self.tokenized_text

        # Split text into segments
        text_segments = []
        for i in range(0, len(tokenized_text), self.text_segment_size):
            text_segments.append(tokenized_text[i:i+self.text_segment_size])

        # Split audio into segments
        audio_segment_len = self.audio_segment_size * group_size * audio_channels
        audio_segments = []
        for i in range(0, len(self.audio), audio_segment_len):
            audio_segments.append(self.audio[i:i+audio_segment_len])

        tokenized_segments = []
        tokenized_segments.append(
            InputSegment(
                text='<|sostm|>',
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.text_zeroemb_idx,
            ),
        )

        eot_tokens = tokenizer(
            "<|eot|>",
            return_tensors="np",
            truncation=True,
            max_length=999999,
            padding=False,
            add_special_tokens=False,
        )["input_ids"][0]
        eot_tokens = jnp.array(eot_tokens, dtype=jnp.int32)

        # Append eot to last text segment
        if len(text_segments) > 0:
            text_segments[-1] = jnp.concatenate([text_segments[-1], eot_tokens])

        length = min(len(text_segments), len(audio_segments))
        for i in range(length):
            text_segment = text_segments[i]
            audio_segment = audio_segments[i]

            tokenized_segments.append(
                InputSegment(
                    tokenized_text=text_segment,
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.text_zeroemb_idx,
                ),
            )
            tokenized_segments.append(
                InputSegment(
                    audio=audio_segment,
                    add_sosp_eosp=False,
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.text_zeroemb_idx,
                ),
            )

        for j in range(length, len(text_segments)):
            tokenized_segments.append(
                InputSegment(
                    tokenized_text=text_segments[j],
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.text_zeroemb_idx,
                ),
            )

        for j in range(length, len(audio_segments)):
            tokenized_segments.append(
                InputSegment(
                    audio=audio_segments[j],
                    add_sosp_eosp=False,
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.text_zeroemb_idx,
                ),
            )

        tokenized_segments.append(
            InputSegment(
                text="<|eostm|>",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.text_zeroemb_idx,
            ),
        )

        input_ids = [
            seg.to_input_id(
                self.tokenizer,
                self.group_size,
                self.audio_channels,
            )
            for seg in tokenized_segments
        ]


        input_ids = jnp.concatenate(input_ids, axis=1).astype(jnp.int64)  # [n_rvq + 1, seqlen]

        return input_ids
