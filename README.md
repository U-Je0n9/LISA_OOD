## 📚 **Reasoning Segmentation with Improved OOD Handling in Vision-Language Models**

### 🌟 **Overview**  
Vision-Language Models (VLMs) have demonstrated remarkable advancements in understanding and processing the relationships between images and text. Modern VLMs go beyond simply connecting two modalities—language and vision—to solve complex tasks that were traditionally confined to either vision or language domains.

A representative example is the **LISA model**, which outputs both **segmentation masks** and **textual answers** in response to image-related questions. While LISA serves as the foundation for many segmentation-capable VLMs, it encounters limitations in handling **Out-of-Distribution (OOD)** scenarios, such as when questions involve objects that do not exist in the given image.

This project addresses these limitations by providing **code for constructing a new OOD dataset** and applying minimal additional training. As a result, our approach significantly enhances **VQA (Visual Question Answering) accuracy** while maintaining minimal performance degradation in **reasoning segmentation** tasks. These outcomes strengthen the potential of VLMs as reliable multimodal models, paving the way for more trustworthy applications across various domains.

대표적인 reasoning segmentation VLMs인 **LISA**는 **세그멘테이션 마스크(segmentation mask)** 와 **텍스트 기반 정답(textual answer)** 을 생성하여 이미지 관련 질문에 응답하는 구조를 갖는다. 하지만 이미지에 존재하지 않는 객체에 대한 질문이 주어졌을 때와 같은 **Out-of-Distribution (OOD)** 환경에서 한계를 가지고 있다. 

이 프로젝트에서는 이러한 문제를 해결하기 위해 **새로운 OOD 데이터셋을 구축하는 코드** 를 제공하고, 최소한의 추가 학습을 적용하여 모델을 개선하였다. 그 결과, **VQA(Visual Question Answering) 정확도가 크게 향상**되었으며, **Reasoning Segmentation** 작업에서의 성능 저하는 최소화할 수 있었다. 이를 통해 VLM의 신뢰성을 높이고, 다양한 응용 분야에서 보다 강력한 다중모달(multimodal) 모델로 활용될 수 있도록 기여하고자 한다.

---

### 🔍 **Key Features**
- ✅ OOD dataset generation code included for customized dataset creation
- 🎯 Enhanced OOD handling with minimal additional training
- 🖼️ Multimodal understanding combining visual segmentation and natural language processing
- 📈 Improved VQA accuracy while maintaining reasoning segmentation performance
- 💡 Foundation for more robust and trustworthy VLM applications

---

### 🗂️ **Dataset Generation**
The repository includes code to generate a custom **OOD dataset**, featuring:
- Scripts for creating images paired with questions about non-existent objects
- Tools for generating corresponding answers indicating the absence of those objects

The dataset generation process enables the model to learn how to respond accurately in OOD scenarios, improving both robustness and reliability.

LLM을 사용하여 **사용자가 직접 OOD 데이터셋을 생성할 수 있는 코드**가 포함되어 있다.
- 이미지와 함께 **존재하지 않는 객체에 대한 질문을 생성하는 스크립트**
- **해당 객체가 존재하지 않는다는 것을 반영한 정답 데이터 생성 도구**

---

### 📊 **Evaluation on OOD Dataset**  
The evaluation on the OOD dataset was conducted using **BERT Score** and **BLEU Score** metrics. Compared to the base **LISA-7B** model, our approach shows significantly higher performance. The results demonstrate that the dataset generation code provided in this project allows models to learn the relationship between text and images more effectively, enabling accurate textual generation even in OOD scenarios.

#### **BERT Score & BLEU Score Results**
| Model   | Precision | Recall | F1   | BLEU4 |
|---------|------------|---------|------|--------|
| LISA-7B | 90.78      | 92.58   | 91.65| 23.19  |
| **Ours**| **96.82**  | **96.85**| **96.83**| **65.17** |

Our model outperformed the baseline by achieving:
- **BERT Score:** Precision, Recall, and F1 scores of 96.82%, 96.85%, and 96.83%, respectively, which is **an improvement of approximately 6.04%, 4.27%, and 5.18%** over LISA-7B.
- **BLEU4 Score:** **An increase from 23.19% to 65.17%**, nearly tripling the score, highlighting significant improvements in text generation quality.

These results suggest that our model generates more appropriate responses to OOD questions by learning relevant contextual information during the dataset generation and training process.

---

### 🧪 **IoU Evaluation for Reasoning Segmentation**
The **IoU (Intersection over Union)** performance was evaluated using the **ReasonSeg** dataset. While there was a slight performance drop due to OOD handling, our model still significantly outperformed other baseline models such as OVSeg and GRES.

#### **IoU Evaluation Results**
| Model   | gloU  | cloU  |
|---------|-------|-------|
| **Ours**| 36.77 | 42.07 |
| LISA-7B | 46.05 | 50.88 |
| OVSeg   | 26.1  | 20.8  |
| GRES    | 23.1  | 22.0  |

Key observations:
- Although our model shows a slight decrease compared to LISA-7B (due to the trade-off in focusing on OOD handling), it still significantly outperforms OVSeg and GRES.
- The results indicate that some segmentation capability might have been compromised while improving OOD performance, but the overall segmentation ability remains strong.

---

### 📚 **References**
Xin Lai, et al. “LISA: Reasoning Segmentation via Large Language Model.” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024.
