## Quickstart
1. Install the library
```bash
python3 -m pip install --user git+https://github.com/thousandvoices/punctuation_corrector.git
```
2. Use the pretrained models
```python
from punctuation_corrector.inference import Corrector

corrector = Corrector.load(
    'https://github.com/thousandvoices/punctuation_corrector/releases/download/v0.1/ru_comma_onnx_quantized.zip')
print(corrector.correct(['Он не понимал светило ли на небе солнце не мог даже определить день сейчас или ночь']))

# If all goes well, the result should be
# ['Он не понимал, светило ли на небе солнце, не мог даже определить, день сейчас или ночь']
```
