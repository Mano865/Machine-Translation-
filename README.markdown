# English to Egyptian Arabic Machine Translation

This project implements a machine translation model to translate English sentences into Egyptian Arabic using the `Helsinki-NLP/opus-mt-en-ar` model from the Hugging Face Transformers library. The model is fine-tuned on the `HeshamHaroon/ArzEn-MultiGenre` dataset, which contains parallel English and Egyptian Arabic text.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Example Translations](#example-translations)
- [Dependencies](#dependencies)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project demonstrates how to:
- Load and preprocess a dataset for machine translation.
- Fine-tune a pre-trained transformer model (`Helsinki-NLP/opus-mt-en-ar`) for English to Egyptian Arabic translation.
- Evaluate the model using metrics like BLEU and ROUGE.
- Perform translations on new English sentences.

The notebook uses Python libraries such as `transformers`, `datasets`, `evaluate`, and `pandas` to handle data processing, model training, and inference.

## Dataset
The dataset used is `HeshamHaroon/ArzEn-MultiGenre`, available on Hugging Face. It contains parallel sentences in English (`ENG`) and Egyptian Arabic (`EGY`). The notebook loads 20,000 rows from the dataset and applies preprocessing steps:
- Removes rows with missing or duplicate values.
- Filters out invalid text (e.g., empty strings, purely numerical text, or special characters).
- Ensures sentences have a minimum length of 2 characters.

Sample data (first 10 rows):
| ENG | EGY |
|-----|-----|
| Already? | ‫لحق؟‬ |
| Sorry to keep you waiting. | ‫معلش يا جماعة أخرتكم.‬ |
| No problem. | ‫لا، ولا يهمك.‬ |
| The system was down. | ‫بس الsystem down.‬ |
| ... | ... |

## Installation
To run this project, you need Python 3.11 or later and the following dependencies. Follow these steps to set up the environment:

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install torch transformers accelerate numpy pandas datasets evaluate rouge_score sacrebleu
   ```

4. **Verify installation**:
   Run the first cell of the notebook to check the versions of key libraries:
   ```
   Torch: 2.6.0+cu124
   Transformers: 4.53.2
   Accelerate: 1.9.0
   Numpy: 2.0.2
   ```

## Usage
1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook Machine_Translation.ipynb
   ```

2. **Run the notebook cells**:
   - Execute the cells sequentially to load the dataset, preprocess the data, fine-tune the model, and test translations.
   - The notebook includes a function `translate_to_egyptian(sentence)` to translate new English sentences into Egyptian Arabic.

3. **Translate new sentences**:
   Use the `translate_to_egyptian` function to translate custom English sentences. Example:
   ```python
   sentence = "How are you doing today?"
   translation = translate_to_egyptian(sentence)
   print(f"EN: {sentence}")
   print(f"EG: {translation}")
   ```

## Model Training
The model is fine-tuned using the `Helsinki-NLP/opus-mt-en-ar` transformer model with the following setup:
- **Tokenizer and Model**: `AutoTokenizer` and `AutoModelForSeq2SeqLM` from Hugging Face.
- **Training Arguments**:
  - Batch size: 16 (per device for training, 8 for evaluation).
  - Learning rate: 2e-5.
  - Number of epochs: 30.
  - Early stopping with patience of 3 epochs.
  - Mixed precision training (`fp16=True`) for efficiency on GPU.
- **Data Preprocessing**:
  - Tokenizes English and Egyptian Arabic sentences with a maximum length of 64 tokens.
  - Splits the dataset into 80% training and 20% validation sets.
- **Evaluation Metrics**:
  - BLEU (via `sacrebleu`) and ROUGE scores are computed during training.
- **Training Output**:
  - The model was trained for 52,230 steps with a training loss of approximately 0.514.
  - Total FLOPs: ~2.66e15.
  - Training runtime: ~7,199 seconds (~2 hours).

## Example Translations
The notebook includes a sample of translations from English to Egyptian Arabic. Below are some examples:
1. EN: How are you doing today?  
   EG: ازيكو النهاردة؟
2. EN: Don't talk any more  
   EG: متتكلميش!
3. EN: I love Egyptian food.  
   EG: أنا بحب الأكل المصري
4. EN: This place is beautiful.  
   EG: ده المكان جميل اوي.
5. EN: Are you coming with us?  
   EG: هتيجي معانا？

## Dependencies
- Python 3.11+
- `torch==2.6.0+cu124`
- `transformers==4.53.2`
- `accelerate==1.9.0`
- `numpy==2.0.2`
- `pandas`
- `datasets`
- `evaluate`
- `rouge_score`
- `sacrebleu`

Install them using:
```bash
pip install torch transformers accelerate numpy pandas datasets evaluate rouge_score sacrebleu
```

## Results
The fine-tuned model achieves reasonable translations for conversational English to Egyptian Arabic, as shown in the example translations. However, some translations may deviate from the expected output due to:
- Dataset noise or inconsistencies.
- Limitations in capturing the full range of Egyptian Arabic dialects.
- Model capacity or training duration.

To improve performance:
- Increase the dataset size beyond 20,000 rows.
- Experiment with hyperparameter tuning (e.g., learning rate, number of beams).
- Use a larger model or extend training with more epochs.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

Please ensure your code follows the project's style and includes relevant tests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.