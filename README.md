# LUNA_Linear_Unified_Nested_Attention
현재 인코더까지만 구현 디코더는 구현 후 업데이트 예정

## Usage
luna_encoder_exmaple.ipynb 참고

## Paper
https://arxiv.org/pdf/2106.01540v2.pdf

## 주요 내용

메타(페이스북)에서 낸 변형된 트랜스포머의 어텐션 구조를 제안한 논문이다.

![image](https://user-images.githubusercontent.com/63130907/200698452-aaa21d9b-04d2-467a-ae53-6913fa199e9a.png)

![image](https://user-images.githubusercontent.com/63130907/200697395-b61f587b-bdb8-484f-8873-b31705b634ca.png)

일반적인 트랜스포머의 어텐션 구조는 len(query_len) * len(key_len)의 구조를 갖게 된다. Self attention의 경우에는 query와 key의 input이 같기 때문에 결국 query_len의 제곱만큼의 계산이 필요하게 된다. 

이러한 연산량에 대한 개선방법으로 Linformer, Performer 등의 방법이 제안 되었지만 연산량에 대한 trade-off로 성능하락이 있는 것으로 보인다. 논문의 저자는 연산량이 제곱이 아니라 선형적으로 필요한 LUNA를 제안한다.

기본적으로는 Alber나 Polyer Encoder에서 나오는 개념과 비슷하게 저차원으로 압축한 후에 다시 복원하는 형식을 가지는데, 위의 그림과 같이 단순히 압축 -> 복원의 구조가 아니라 각각의 요소들을 조합하는 구조를 갖고 있다.

트랜스포머의 Multi-head attention을 2개로 나눠서 수행을 하고, 기존의 query, key, value에 더해 고정된 길이의 벡터 p를 이용하여 어텐션을 수행한다. 여기서 p는 projection의 p라고 생각해도 될 것 같다.

구조의 차이는 Encoder를 예를 들면 아래와 같다.
구조 내에서도 여러가지 선택지가 있지만 여기서는 최대한 기존의 트랜스포머와 비슷한 구조로 설명을 하겠다.

#### 기본 트랜스포머의 Encoder 구조
embedded = word embedding * scale + positional encoding     
embedded = query = key = value   
   
첫 input : embedded   
이후 input : 아래 Encoder_layer의 output   
----- Encoder_layer_loop -----   
attention_outputs, attention_weights = MutlheadAttention(query, key, value, attention_mask)   
attention_outputs = Norm(attention_outputs + embedded)   
   
encoder_outputs = Norm(Feedforward(attention_outputs) + attention_outputs)   

return encoder_outputs, attention_weights   

#### LUNA의 Enocder 구조
embedded = word embedding * scale + positional encoding     
embedded = query = key = value   
   
p = nn.Parameter(torch.tensor(설정한 p의 길이, d_model))   
p = p * scale + positional encoding # seq_len, d_model   
p = p.expand(batch_size, seq_len, d_model)   
   
첫 input : embedded, p   
이후 input : Encoder output의 Yx, Yp   
----- Encoder_layer_loop -----   
Yp = MutlheadAttention(p, key, value, attention_mask)   
Yx = MutlheadAttention(query, Yp, Yp) # p의 경우에는 padding이 없으므로 어텐션 마스크 없음   
    
Yp = Norm(Yp + p)   
Yx = Norm(Yx + embedded)   
Yx = Norm(Feedforward(Yx) + Yx)   

return Yp, Yx

또한 NLU Task 수행을 위해서 보통 마지막 레이어의 outputs중 CLS 토큰 위치의 값을 사용하는데, LUNA 저자는 p역시 문맥의 의미를 담고 있다고 보고 마지막 레이어의 p를 mean pooling하여 분류 Task를 수행하였는데, 같은 조건일 경우 CLS를 사용할 때보다 소폭이지만 더 좋은 성능을 보였다.

## 인사이트

아이디어 자체는 메타의 Albert, Poly Encoder나 마이크로소프트의 LoRA가 사용하는 방식과 비슷한 것 같다. 하지만 연산을 나눈 후에 각 파트들을 조합하여 사용한다는 점에서 차이가 있는 것 같다. 실제로 성능이 어느정도로 차이가 나는지는 추후 실험을 해봐야할 것 같다. 트랜스포머가 나온지도 꽤 되었지만 어텐션이나 트랜스포머의 구조에 대해서 더 생각해볼 필요는 있을 것 같다. 그동안 여러가지 트랜스포머 변형 모델들이 나왔지만 초거대언어모델로 만들 경우 성능은 결국 바닐라 트랜스포머가 좋았다는 논문(Scaling Laws vs Model Architectures: How does Inductive Bias Influence Scaling?)에는 Luna 관련 실험이 없었기 때문에 관련 실험을 해봐야할 것 같다. 이러한 구조들을 볼수록 아직도 갈 길이 멀고, 기존의 방식이 정말 효율적인지에 다시 생각해보게 되는 것 같다.

## Reference
https://paperswithcode.com/paper/luna-linear-unified-nested-attention

https://github.com/XuezheMax/fairseq-apollo

https://github.com/sooftware/luna-transformer
