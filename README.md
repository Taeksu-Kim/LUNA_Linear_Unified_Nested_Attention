# LUNA_Linear_Unified_Nested_Attention

## Usage
LUNA_example.ipynb 참고

## Paper
https://arxiv.org/pdf/2106.01540v2.pdf

## 주요 내용

메타(페이스북)에서 낸 제곱 연산을 갖는 트랜스포머의 구조적 문제를 Low rank 방식으로 해결하는 것을 제안한 논문이다.

![image](https://user-images.githubusercontent.com/63130907/200698452-aaa21d9b-04d2-467a-ae53-6913fa199e9a.png)

![image](https://user-images.githubusercontent.com/63130907/200697395-b61f587b-bdb8-484f-8873-b31705b634ca.png)

고정된 길이를 갖는 p를 이용하여 제곱 연산을 피한다. 기존의 Multi-head attention을 pack, unpack attention으로 나눠서 수행한다.
이때 p는 nn.Parameter로 만든 [p 길이, 임베딩 차원]의 크기의 텐서이다.

## 실험 결과

전체적으로 한번 하는 연산을 나눠서 수행하고 거기에 학습 가능한 파라미터들이 추가되기 때문에 파라미터수는 증가한다.

LUNA를 구조를 적용할 경우 Encoder모델의 파라미터수는 증가하지만 인풋 길이가 길어질수록 속도가 빨라지는 것을 확인할 수 있었다.  
Encoder-Decoder 모델의 경우에는 파라미터수 증가와 함께 속도도 더 오래걸리는 것을 확인할 수 있었다. 이유는 디코더의 셀프 어텐션 구조 문제 때문인 것 같다.   
디코더의 경우에는 인코더와 어텐션 마스크가 다르게 설정되는데, subsquent mask를 추가해서 현재 위치 뒤의 토큰들의 정보를 확인하지 못하는 채로 어텐션을 수행하게 된다.   
때문에 n*n 매트릭스를 통해 마스크가 적용돼야하는데, p를 이용할 경우 이를 적용시키는 것이 꽤 복잡해지게 된다.

linformer를 참고하여 디코더의 self attention으로 LunaCausalAttention을 제안하였는데, 결국에는 가운데에 p가 끼어드는 것으로 인해서 오히려 연산량이나 속도가 더 증가하게 되는 것 같다. 인코더도 결국 하나의 Multi-head attention을 나눠서 수행하게 되기는 하지만 구조가 그렇게 복잡해지지는 않아 입력 길이가 길어질수록 의도한 결과가 잘 나오는데, 디코더의 경우에는 셀프 어텐션에서 행렬곱의 수도 늘어나고 텐서 조작이 늘어나 바닐라 트랜스포머에 비해 더 시간이 오래 걸리는 결과가 나타났다. 

결과의 성능에 대해서는 실험을 더 해봐야 할 것 같다. linformer도 그렇고 Low rank 논문에서 이야기하는 바와 같이 일정 이상의 크기라면 저차원으로 축소 시켰다 복원하더라도 정보량의 손실이 생각보다 적다는 것을 고려하면 나쁘지 않은 방식 같다. 하지만 역시 디코더의 경우에는 구조적인 문제가 있어서 성능이 더 좋은 것이 아니라면 굳이 디코더에도 이 구조를 적용할 필요는 없을 것 같다.

## 인사이트

그동안 여러가지 트랜스포머 변형 모델들이 나왔지만 초거대언어모델로 만들 경우 성능은 결국 바닐라 트랜스포머가 좋았다는 논문(Scaling Laws vs Model Architectures: How does Inductive Bias Influence Scaling?)에는 Luna 관련 실험이 없지만 디코더 구조 문제도 있고, 마찬가지로 큰 개선점을 보이기는 어려울 것 같다.  

트랜스포머의 구조 개선에 대한 여러가지 연구가 진행되고 있지만 들어가는 노력에 비해 결과는 애매한 경우가 많은 것 같다. 하지만 Low rank 논문들에서 얘기하는 것과 같이 실제로 결과에 결정적인 영향을 주는 파라미터들은 생각보다 그 수가 많지 않을 수도 있고, 좀 더 구조적인 개선이나 새로운 구조를 통해 개선할 여지는 아직도 충분히 있을 것 같다. 

LunaCausalAttention에서는 attention을 구할 때 직접적인 어텐션 연산은 아니지만 softmax 대신에 softplus를 사용했는데, 생각해보면 softmax의 경우에는 결국 총합을 1로 만들기 때문에 구조적인 한계가 있을 수 있다. Low rank같은 경우에도 저차원으로 projection하는 것은 사람이 전문보다 요약문을 통해 더 주의를 기울여야하는 부분을 잘 알 수 있는 것과 같이 직관적으로 생각해도 효과가 있고, 논문들에서도 어느정도 효과가 확인이 되었다. 개인적으로는 이 projcetion을 입력 길이보다는 임베딩 차원등 다른 쪽으로하는 것이 나을 수 있겠다는 생각이 들었는데, 디코더의 auto rgressive적인 측면을 적용하기가 어렵기 때문이다. 이것도 수학적인 다른 트릭을 통해 극복할 수 있을지도 모르겠지만 여러모로 아직도 갈 길이 멀다는 생각이 들었다. 

## Reference
https://paperswithcode.com/paper/luna-linear-unified-nested-attention

https://github.com/XuezheMax/fairseq-apollo

https://github.com/sooftware/luna-transformer
