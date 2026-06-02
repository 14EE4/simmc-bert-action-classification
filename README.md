# SIMMC BERT Action Classification

SIMMC 형식의 대화 데이터에서 사용자 발화(`transcript`)를 기반으로 assistant action label(`action_id`)을 분류하는 BERT 기반 실험 노트북입니다. K-Fold 검증과 Optuna 하이퍼파라미터 탐색을 사용하며, Kaggle과 Google Colab 환경을 모두 지원합니다.

노트북의 첫 셀에서 실행 환경별 데이터 경로와 모델 저장/로드 경로를 한 번에 설정할 수 있습니다.

## 파일 구성

- `kaggle-colab-bert-action-kfold-optuna-optimized.ipynb`: Kaggle/Colab 공용 BERT action classification 실험 노트북

## 필요한 데이터

노트북은 아래 CSV 파일을 사용합니다.

| 파일 | 용도 |
| --- | --- |
| `Origin_Train_Data_10k.csv` | 원본 훈련 데이터 |
| `Train_Data_10k_2.csv` | LLM 증강 훈련 데이터 |
| `Middle_Test_Data.csv` | 검증 및 조기 종료용 데이터 |
| `Final_Test_Data.csv` | 최종 평가용 테스트 데이터 |

기본적으로 아래 컬럼을 기대합니다.

- `transcript`: 사용자 발화 텍스트
- `action_id`: 분류 대상 action class
- `system_transcript`: 일부 평가/데이터 구성 셀에서 사용하는 선택적 시스템 문맥 컬럼

## 실행 환경 설정

노트북을 열고 첫 셀의 `RUN_ENV` 값을 설정합니다.

```python
RUN_ENV = "kaggle"  # "colab" 또는 "kaggle"
```

Colab에서 실행하는 경우:

- `RUN_ENV = "colab"`로 설정합니다.
- 첫 셀에서 `drive.mount('/content/drive')`로 Google Drive를 마운트합니다.
- 기본 데이터 경로는 `/content/drive/MyDrive/AI_Agent_Project`입니다.
- 기본 모델 저장 경로는 `/content/drive/MyDrive/AI_Agent_Project/Saved_Models`입니다.

Kaggle에서 실행하는 경우:

- `RUN_ENV = "kaggle"`로 설정합니다.
- 기본 데이터 경로는 `/kaggle/input/datasets/pyeong85732/simmc-csv`입니다.
- Kaggle Dataset의 실제 경로가 다르면 첫 셀의 `KAGGLE_INPUT_BASE_PATH`를 수정합니다.
- 기본 모델 저장 경로는 `/kaggle/working/Saved_Models`입니다.

## 결과물

학습된 모델 체크포인트는 실행 환경에 맞는 `Saved_Models` 디렉터리에 저장됩니다. 뒤쪽의 평가 및 시각화 셀은 첫 셀에서 정의한 `output_dir`, `save_dir`, `TARGET_DF_PATH`, `resolve_model_path()`를 사용해 저장된 모델을 찾습니다.

## 참고

- 이 폴더는 원본 SIMMC 저장소 전체가 아니라, SIMMC 기반 BERT action classification 실험을 위한 별도 작업 폴더입니다.
- 데이터셋 라이선스, 사용 조건, baseline 관련 세부 내용은 원본 SIMMC 저장소를 확인하세요.

## 데이터 라이선스

`data/` 폴더에 업로드하는 전처리 CSV 파일은 SIMMC 데이터셋을 기반으로 가공한 데이터입니다. 따라서 비상업적 연구 및 교육 목적으로만 공유하며, 원본 SIMMC 데이터셋과 동일하게 CC BY-NC-SA 4.0 라이선스 조건을 따릅니다.

전처리 CSV를 사용할 때는 아래 사항을 지켜 주세요.

- 원본 데이터가 SIMMC 데이터셋에서 파생되었음을 표시합니다.
- 전처리 또는 수정된 데이터임을 표시합니다.
- 상업적 목적으로 사용하지 않습니다.
- 동일한 CC BY-NC-SA 4.0 조건으로 공유합니다.
- 아래 SIMMC 논문을 인용합니다.

원본 SIMMC 저장소: <https://github.com/facebookresearch/simmc>

## 인용

이 프로젝트는 SIMMC 기반 데이터를 사용합니다. 이 데이터를 사용한 실험 결과를 공개하거나 논문/보고서에 활용하는 경우 아래 SIMMC 논문을 인용해 주세요.

```bibtex
@article{moon2020situated,
  title={Situated and Interactive Multimodal Conversations},
  author={Moon, Seungwhan and Kottur, Satwik and Crook, Paul A and De, Ankita and Poddar, Shivani and Levin, Theodore and Whitney, David and Difranco, Daniel and Beirami, Ahmad and Cho, Eunjoon and Subba, Rajen and Geramifard, Alborz},
  journal={arXiv preprint arXiv:2006.01460},
  year={2020}
}

@article{crook2019simmc,
  title={SIMMC: Situated Interactive Multi-Modal Conversational Data Collection And Evaluation Platform},
  author={Crook, Paul A and Poddar, Shivani and De, Ankita and Shafi, Semir and Whitney, David and Geramifard, Alborz and Subba, Rajen},
  journal={arXiv preprint arXiv:1911.02690},
  year={2019}
}
```

## 라이선스

SIMMC 데이터셋은 [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) 라이선스로 배포됩니다. 전체 라이선스와 사용 조건은 원본 SIMMC 저장소를 확인하세요.
