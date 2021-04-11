# DeepLearning
## Image Classification

- cifar10 데이터사용
- 내가 찍은 사진을 자동차, 비행기, 고양이 등의 카테고리로 분류하는 프로젝트
- keras cnn 사용

## Natural Language Processing

- 학습 교재에서 제공하는 기사 크롤링 데이터 사용 (text 폴더)
- 새로운 문장을 입력했을 때 사회, 경제, 정치 등의 카테고리로 분류하는 기능이 있는 웹 페이지를 구현하는 프로젝트
- keras mlp 사용
- python flask 프레임워크, 형태소 분석 okt 라이브러리 사용

1. [tfidf.py](http://tfidf.py) 에서 TF-IDF를 사용하여 text 폴더 안에 있는 인터넷기사에 사용된 단어를 수치로 변환하고, 그 결과를 genre.pickle로 만듦.(단어 사전 생성) 
2.  train_db.py 에서는 genre.pickle을 사용하여 나이브베이즈로 모델을 생성하고 성능을 측정해봄.
3. train_mlp.py에서는 genre.pickle을 mlp를 사용하여 모델을 생성하고 성능을 측정해봄.
4. mytest.py에서는 mlp 기사 분류 모델에 학습에 사용되지 않은 기사문을 입력하여 기사의 카테고리를 분류해봄.
