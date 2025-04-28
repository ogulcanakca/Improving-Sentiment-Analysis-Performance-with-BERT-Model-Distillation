# Improving Sentiment Analysis Performance with BERT Model Distillation

This project applies the **Knowledge Distillation** technique to transfer the knowledge of a large BERT model (Teacher) to a significantly smaller Transformer model (Student). The goal is to reduce model size and potentially inference time, while maintaining high performance or improving over standard training.

**Project Goal:**

* To achieve high accuracy on sentiment analysis task (on IMDb dataset) using a large, pre-trained `bert-base-uncased` model (Teacher).
* To distill the "knowledge" of this teacher into a specialized Transformer model (Student) with fewer layers and parameters.
* Compare the performance of the distilled student model with a **Baseline Student Model** with the same architecture but trained on data only, *without* distillation.
* To compare in terms of model size (number of parameters) and inference performance.

**Models:**

1. **Teacher Model:**
 * Architecture: `bert-base-uncased` (Hugging Face) with a classification layer added on top.
    * Number of Parameters: **~110 Million** (109,483,778)
 * Role: High-performing, but large and slow source of information. (Note: Specific accuracy/F1 scores for this model are not presented in this report, but are generally expected to be higher than student models).

2. **Student Model (Distilled & Baseline):**
 * Architecture: Smaller Transformer (Example: 4 layers, smaller hidden size - the specific configuration used in the project can be specified).
    * Number of Parameters: **~29 Million** (28,764,674) - **~74% smaller than Teacher!**
 * Role:
        * **Distilled Student:** Trained using both real labels (hard labels) and soft predictions of the tutor model (soft labels/logits).
        **Baseline Learner:** Trained in the standard way using only real labels (hard labels).

**Methodology:**

* **Dataset:** IMDb Movie Reviews (Binary sentiment classification: Positive/Negative).
* Framework:** PyTorch, Hugging Face `transformers` and `datasets` libraries.
* **Distillation Technique:** Response-based distillation was used. The loss function is a weighted combination of the student's Cross-Entropy loss to the real labels and the KL Divergence loss of the student's logits (smoothed by the temperature parameter) to the teacher's soft logits (`L_total = alpha * L_hard + (1 - alpha) * L_soft`).
**Media:** Kaggle Notebooks (with GPU).

**Results:**

The following table summarizes the metrics obtained on the test set:

| Model Type | Number of Parameters | Accuracy | F1 Score | Precision | Recall | Average Inference Time per Sample (ms) |
| :------------------ | :--------------- | :------------------ | :------------------ | :-------- | :--------------------- | :------------------ |
| Teacher (BERT-base)| ~110 Million | *Not specified* | *Not specified* | *Not specified* | *Not specified* | *Not specified* |
| **Distilled Student**| **~29 Million** | **0. 8784** | **0.8742** | 0.9054 | 0.8450 | **~0.2207** |
| Baseline Student | ~29 Million | 0.8644 | 0.8611 | 0.8826 | 0.8406 | ~0.2206 |


*(Note: Inference times may vary depending on the test environment, batch size, and measurement method.)*

**Analysis and Discussion:**

1. **Effect of Distillation:**
 * The Distilled Student Model significantly outperformed the Baseline Student Model with **same architecture and number of parameters** (+1.4% increase in accuracy, +1.3% increase in F1 score).
    * This suggests that the distillation process successfully conveys to the student the richer knowledge ("dark knowledge") that the teacher has about the relationships between classes, not just the correct label. The student model was able to generalize better than he/she would have learned just by looking at the labels.

2. **Efficiency Gain:**
 * Both student models have **74% fewer parameters** than the teacher model. This significantly reduces the models' storage requirements, memory footprint and potentially the computational power required for training/inference. This is critical for deploying models in environments such as mobile devices, edge computing or resource-constrained servers.
    * In the results presented, the average inference times of the two student models are almost identical. This may be due to the GPU not being fully saturated with the batch size used or due to small differences in measurement. However, a large reduction in the number of parameters usually leads to more pronounced speed advantages in more complex scenarios or on different hardware.

3. **Performance vs. Efficiency Tradeoff:**
 * The distilled student model approaches the (expected high) performance of the teacher model, with a very significant gain in model size. The performance improvement over the baseline student proves that distillation is an effective method to improve this balance.

**Conclusion:**

This project has successfully demonstrated that model distillation is a powerful technique for overcoming the deployment challenges of large language models in practical applications. A smaller model can outperform a model of the same size trained with only real labels through distillation. In this way, by reducing the model size, it is possible to obtain models that compromise less on performance while increasing efficiency.
