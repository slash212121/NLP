# NLP
Main concepts of NLP and how to implement them

## Word Meaning Representation and Encoding(단어의미 표현 및 인코딩)
- Language has several layers to analyze, and thus is highly complex.(언어는 다양한 단계및 측면에서 분석하여야 하기 때문에, 상당히 복잡한 영역이다)

![alt text](https://upload.wikimedia.org/wikipedia/commons/7/79/Major_levels_of_linguistic_structure.svg)

- phonetics(음성론):언어자체가 어떻게 들리는지에 초점을 맞춤
- phonology(음운론):언어가 소리를 배합하는 체계에 초점을 맞춤
- morphology(형태론):단어의 구조에 초점을 맞춤
- syntax(통사론):문법 구조에 초점을 맞춤
- semantics(의미론):문장 및 표현의 표면적 의미에 초점을 맞춤
- pragmatics(화용론):문장 및 표현의 맥락에 따른 의미에 초점을 맞춤.(ex.A:(시간약속 늦은 친구에게)너 지금 몇시인지 알아? B:미안미안, 다음부터 안 늦을게!)

In order to fully grasp the concept of language one must be aware of all these levels of language, which is exactly why NLP is a difficult field in computer sciences. The main problem is: how do you get a computer to understand these highly complex layers of language?(언어를 제대로 이해하기 위해서는, 언어의 모든단계를 이해해야 할 필요가 있다. 하지만, 문제가 하나 발생한다: 어떻게 컴퓨터한테 이 모든 단계를 이해하게 만들지?)
Not only that, every language has its own set of unique rules. How can a computer learn these specific rules? These are some of the main tasks that the NLP field must tackle.(뿐만 아니라 모든 언어마다 각자 고유의 규칙이 있다. 어떻게 하면 컴퓨터가 이런 구체적인 규칙들을 배울 수 있는 것일까? 자연어처리 영역에서 이런 문제들을 주로 다루고 있다.)

First things first, how do we make a computer understand words? To be more specific, how does a computer see representations of words? This whole process is something called encoding.(우선 우리가 생각해야 할 것은, 컴퓨터가 어떻게 단어를 이해하는 것인가이다. 좀더 구체적으로 말하면, 컴퓨터는 인간의 단어를 어떻게 인식하는 것일까? 이런 과정을 바로 ***인코딩*** 이라 말한다)

The following methods and models have different approaches of encoding. They are listed in this order as the models progress and improve they are built up on the previous models and attempt to modify any flaws that the previous models may have had.(다음 모델들은 각자 다른 방식으로 인코딩을 하게 된다. 자연어처리의 모델들은 순서가 중요한데, 그 이유는 지난 모델들이 안고있는 문제점들을 개선해나가는 과정에서 모델들이 발전해오기 떄문이다.)

### 1. One-Hot Encoding(원핫 인코딩)
- Considered one of the most basic forms of encoding(인코딩의 기본형으로 간주됨)
- Any repeated words in a corpus are eliminated, thus making each unique word appear once. Note that words with different forms are considered different words. For example 'cats' and 'cat' are considered different words.(단어들의 집합에서 중복되는 단어들을 제외하여, 고유의 단어 하나만 남긴다. 한가지 주의 할것은, 완전히 똑같은 형태의 단어가 아니면 다른 단어로 처리된다는 것이다. 예를 들면 'cat'와 복수형의 'cats'은 다른 단어로 취급한다.
- Each unique word is given a specific index. To represent a specific word, that word's index is represented with a 1, while the rest are represented as 0.(각 고유의 단어마다 인덱스가 배정이 된다. 특정 단어를 표현하기 위해, 해당되는 단어의 인덱스는 1로, 나머지는 0으로 표현이 된다.)
![alt text](https://miro.medium.com/max/674/1*9ZuDXoc2ek-GfHE2esty5A.png)

This method is highly simplistic and easy to understand. However this method poses many flaws as well.(원핫인코딩은 상당히 단순하고 이해하기 쉬운 개념이다. 그러나, 많은 한계를 지니고 있다.)

- The size of a one-hot vector is dependent on the absolute size of the corpus. The larger the size of the text is, the larger the encoded vector becomes. The size of the vector will inevitably cause computing problems when it comes to speed.(원핫벡터의 크기는 코퍼스 크기에 절대적으로 의존한다. 텍스트양이 많아지면 많아질수록, 인코딩된 벡터의 크기 또한 늘어난다. 이는 결론적으로 연산의 속도가 느려지는데 크게 기여하게 된다.)
- One-Hot embeddings have another flaw in which they are not able to capture nuances between words as they are only comprised of 1's and 0's. To elaborate, a 'cat' and a 'tiger' can be considered as having many similarities. They both are felines, they have whiskers, etc. Yet, when your vectors look like this:
```
cat = [0,0,1,0]
tiger = [0,1,0,0]
```
it is very difficult to tell what similarities or differences these embeddings have.(원핫코딩의 치명적인 단점은, 1과 0만으로 구성되어 있기 때문에 단어간의 공통점, 차이점을 표현하기 매우 어렵다는 것이다. '고양이'와 '호랑이'는 생각해보면 고양이족인데다 생긴 것 또한 닮은 것들이 많아, 이런 단어간 유사함이 존재함에도, 단어의 인덱스에만 의존하여 벡터를 만들었기 떄문에 이런 내포되어 있는 정보를 표현하기 힘들다.)

### 2. Count Based Representation-Statistical Language Models(SLM), N-Grams, Bag of Words(BoW)(카운트 기반 표현기법-통계적 언어모델, N-Gram, BoW)
- SLM: Models use conditional probability, mainly the words used before a target word.(통계적 언어모델들은 흔히 지정한 단어 이전에 오는 단어들바탕으로 조건부 확률을 활용하여 그 단어가 나올 확률을 측정한다.)
![alt text](https://yqintl.alicdn.com/b52c26fcd63d66b48698decdcf1aad81ac8b5805.png)





