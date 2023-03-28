import numpy as np
from transformers import AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import random
import torch
from torch.utils.data import DataLoader 

class RobertaPredictor:
    """
    Class RobertaPredictor
    A custom estimator and transformer that wraps the RoBERTa model from Hugging Face's Transformers library
    for sequence classification tasks. This class inherits from BaseEstimator and TransformerMixin classes
    from scikit-learn, enabling it to be used in scikit-learn pipelines.

    Attributes:
    ----------
    model (str): The name of the pretrained RoBERTa model to use. Defaults to 'cross-encoder/nli-roberta-base'.
    train_batch_size (int): The batch size for training. Defaults to 8.
    eval_batch_size (int): The batch size for evaluation. Defaults to 32.
    epochs (int): The number of training epochs. Defaults to 1.
    warmup_steps (float): The ratio of the total training steps to use for warm-up. Defaults to 0.1.
    eval_steps (int): The number of training steps between evaluations. Defaults to 500.
    metric_for_best_model (str): The metric to use for early stopping and selecting the best model. Defaults to 'eval_loss'.
    output_dir (str): The directory to save the model's output. Defaults to './results'.
    load_best_model_at_end (bool): Whether to load the best model found during training. Defaults to True.
    random_seed (int): The random seed for reproducibility. Defaults to 42.

    Methods:
    --------
    fit(self, X, y): Trains the RoBERTa model using the provided training data.
    predict(self, X, y=None): Makes predictions using the trained RoBERTa model.

    Example:
    --------
    from model import RobertaPredictor
    transformer_trainer = RobertaPredictor('cross-encoder/nli-roberta-base')
    transformer_trainer.fit(train_set, validation_set)
    predicted_labels, accuracy = transformer_trainer.predict(test_set, test_set['label'])
    print(f'The accuracy of the model is {accuracy}')
    """
    def __init__(self, model:str='cross-encoder/nli-roberta-base', train_batch_size:int=8, eval_batch_size:int=32, epochs:int=1, warmup_steps:int=0.1,
                 eval_steps:int=500, metric_for_best_model:str='eval_loss', output_dir:str='./results',
                 load_best_model_at_end:bool=True, random_seed:int=42):
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.eval_steps = eval_steps
        self.metric_for_best_model = metric_for_best_model
        self.output_dir = output_dir
        self.load_best_model_at_end = load_best_model_at_end
        self.random_seed = random_seed
        self.config = AutoConfig.from_pretrained(model)
    
    def fit(self, train_data, val_data):
        # Sample the training data
        # random.seed(self.random_seed)
        train_data_sample = train_data #.shuffle(seed=self.random_seed).select(range(10000))

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy='steps',
            eval_steps=self.eval_steps,
            load_best_model_at_end=self.load_best_model_at_end,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            num_train_epochs=self.epochs,
            metric_for_best_model=self.metric_for_best_model,
            warmup_steps=self.warmup_steps
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data_sample,
            eval_dataset=val_data,
            compute_metrics=lambda pred: {"accuracy": accuracy_score(
                pred.label_ids, pred.predictions.argmax(axis=1))}
        )

        self.trained_model = self.trainer.train()
        return self

    def predict(self, X):

        # On a eu quelques problèmes avec les etiquettes des label, les chiffres ne correspondent plus après avoir 
        predictions_mapping = self.config.id2label
        label_mapping = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        
        predictions = self.trainer.predict(X)
        
        predicted_labels = np.argmax(predictions.predictions, axis=1)
        accuracy = accuracy_score(
            [sample["labels"] for sample in X],
            [label for label in predicted_labels],
        )
        
        return predicted_labels, accuracy
