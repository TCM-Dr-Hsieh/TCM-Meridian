"""Private worker: local weights and PCM only, no network server."""
import argparse
import base64
from contextlib import redirect_stdout
import json
import sys
import traceback


def send(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def describe_error(exc):
    text = str(exc)
    if 'out of memory' in text.lower():
        return 'ASR 顯存不足。請在 LM Studio 釋放不使用的模型，或將 ASR 運算裝置改為 CPU。'
    return f'{type(exc).__name__}: {text}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--aligner-dir', required=True)
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    args = parser.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            import numpy as np
            import torch
            from qwen_asr import Qwen3ASRModel
            if args.device == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA 不可用；請安裝 CUDA 版 PyTorch，或將運算裝置改為 CPU。')
            device = 'cuda:0' if args.device != 'cpu' and torch.cuda.is_available() else 'cpu'
            torch.set_num_threads(8)
            model = Qwen3ASRModel.from_pretrained(
                args.model_dir, device_map=device,
                forced_aligner=args.aligner_dir,
                forced_aligner_kwargs={
                    'device_map': device,
                    'dtype': torch.bfloat16 if device.startswith('cuda') else torch.float32,
                    'attn_implementation': 'sdpa', 'local_files_only': True,
                },
                dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32,
                attn_implementation='sdpa', local_files_only=True,
                max_inference_batch_size=1, max_new_tokens=512)
        send({'type': 'ready', 'device': device, 'model_dir': args.model_dir, 'aligner_dir': args.aligner_dir})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                samples = np.frombuffer(base64.b64decode(request['audio']), dtype='<f4').copy()
                with redirect_stdout(sys.stderr), torch.inference_mode():
                    results = model.transcribe(audio=(samples, 16000), language=None,
                                               context=request.get('context', ''),
                                               return_time_stamps=request.get('align', False))
                result = results[0]
                items = ([{'text': item.text, 'start': item.start_time, 'end': item.end_time}
                          for item in result.time_stamps.items] if result.time_stamps else [])
                send({'text': result.text, 'language': result.language, 'items': items})
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                send({'error': describe_error(exc)})
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        send({'error': describe_error(exc)})
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
