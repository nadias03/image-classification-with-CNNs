import numpy as np
import torch

class EnsembleVoter:

    VOTING_OPTIONS = ("hard", "soft")
    
    def __init__(self, models: dict, device: str = None, voting: str = "soft"):
        if voting not in self.VOTING_OPTIONS:
            raise ValueError(f"Voting must be one of {self.VOTING_OPTIONS}")
        if len(models) != 3:
            raise ValueError(f"Need 3 models, got {len(models)}")

        self.models = models
        self.voting = voting
        self.device = device

        for model in self.models.values():
            model.to(self.device)
            model.eval()

    def predict(self, loader):
        if self.voting == "hard":
            return self._hard_voting(loader)
        return self._soft_voting(loader)

    def eval(self):
        for model in self.models.values():
            model.eval()
    
    @torch.no_grad()
    def _hard_voting(self, loader):
        all_predictions = []

        for inputs, targets in loader:
            inputs = inputs.to(self.device).float()
            targets = targets.to(self.device)

            batch_predictions = []
            batch_probas = []

            for model in self.models.values():
                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                probas = torch.softmax(outputs, dim=1)
                batch_predictions.append(predictions)
                batch_probas.append(probas)
            
            #shape: (num_models, batch_size)
            stacked = torch.stack(batch_predictions)

            majority_predictions = []
            for i in range(stacked.shape[1]):
                obs_preds = stacked[:, i]  # predictions from each model for this observation
                values, counts = torch.unique(obs_preds, return_counts=True)
                
                if counts.max() > 1:
                    # majority exists
                    majority_predictions.append(values[counts.argmax()])
                else:
                    # no majority — fallback: pick class with highest avg probability
                    avg_proba = torch.stack([bp[i] for bp in batch_probas]).mean(dim=0)
                    majority_predictions.append(torch.argmax(avg_proba))
            
            final = torch.stack(majority_predictions)
            all_predictions.append(final.cpu())
    
        return torch.cat(all_predictions)

    
    @torch.no_grad()
    def _soft_voting(self, loader):
        all_predictions = []

        for inputs, targets in loader:
            inputs = inputs.to(self.device).float()
            targets = targets.to(self.device)

            proba_sum = None

            for model in self.models.values():
                outputs = model(inputs)
                proba_predictions = torch.softmax(outputs, dim=1)

                if proba_sum is None:
                    proba_sum = proba_predictions
                else:
                    proba_sum += proba_predictions
            
            avg_proba_predictions = proba_sum / len(self.models)
            predictions = torch.argmax(avg_proba_predictions, dim=1)
            all_predictions.append(predictions.cpu())

        return torch.cat(all_predictions)






    