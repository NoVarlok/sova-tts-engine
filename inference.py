# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import os
import pandas as pd
import torch

from hparams import create_hparams
from model import load_model

from tps import Handler, cleaners
# from tps import prob2bool, symbols, cleaners


def parse_input_csv(path):
    df = pd.read_csv(path, sep='|', header=None)
    df.columns = ['wav', 'text']
    input_sentences = df['text'].tolist()
    input_names = [name[:-4] for name in df['wav'].tolist()]
    return input_sentences, input_names


def get_text(text, mask_stress, mask_phonemes, text_handler):
    preprocessed_text = text_handler.process_text(
        text, cleaners.light_punctuation_cleaners, None, False,
        mask_stress=mask_stress, mask_phonemes=mask_phonemes
    )
    preprocessed_text = text_handler.check_eos(preprocessed_text)
    text_vector = text_handler.text2vec(preprocessed_text)

    text_tensor = torch.IntTensor(text_vector)
    return text_tensor


def main(input_sentences, input_names, tacotron_path, hparams, use_basic_handler, charset, mask_stress, mask_phonemes, save_dir):
    tacotron = load_model(hparams, False)
    checkpoint_dict = torch.load(tacotron_path, map_location="cpu")
    tacotron.load_state_dict(checkpoint_dict["state_dict"])
    tacotron.cuda().eval()
    print('Model is loaded')

    if use_basic_handler:
        text_handler = Handler(charset)
    else:
        text_handler = Handler.from_charset(charset, data_dir="data", silent=True)

    with torch.no_grad():
        for input_sentence, input_name in zip(input_sentences, input_names):
            processed_sentence = get_text(input_sentence, mask_stress, mask_phonemes, text_handler)
            processed_sentence = torch.unsqueeze(processed_sentence, 0)
            processed_sentence = processed_sentence.cuda()
            mel_outputs, mel_outputs_postnet, gates, alignments = tacotron.inference(processed_sentence)
            mel_save_path = os.path.join(save_dir, f'{input_name}.pt')
            print(mel_save_path)
            torch.save(mel_outputs_postnet[0], mel_save_path)


if __name__ == "__main__":
    model_checkpoint_path = '/home/lyakhtin/repos/tts/results/RuDevices/tacotron2/tacotron-gst-mar-24/checkpoint_186000'
    hparams_path = '/home/lyakhtin/repos/tts/results/RuDevices/tacotron2/tacotron-gst-mar-24/hparams.yaml'
    save_dir = '/home/lyakhtin/repos/tts/results/RuDevices/inference_test/test_mels'
    test_csv = '/home/lyakhtin/repos/tts/datasets/RuDevices/tacotron_inference_short.csv'
    hparams = create_hparams(hparams_path)

    input_sentences, input_names = parse_input_csv(test_csv)

    main(input_sentences=input_sentences,
         input_names=input_names,
         hparams=hparams,
         tacotron_path=model_checkpoint_path,
         use_basic_handler=True,
         charset='ru',
         mask_stress=False,
         mask_phonemes=True,
         save_dir=save_dir)
