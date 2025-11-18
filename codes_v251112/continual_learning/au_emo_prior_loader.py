"""
AU-EMO Prior Loader

This module loads and processes the AU-EMO prior knowledge from the RAF-DB materials,
converting it into the format expected by all continual learning frameworks.

The RAF-DB provides:
- ex_au: [17 emotions, 26 AUs] matrix representing P(AU | EMO)
- 17 emotions: 6 basic + 11 compound
- 26 AUs from RAF-DB AU annotation

Our frameworks use:
- 23 custom AUs (subset/mapping of RAF AUs)
- Variable number of emotions depending on task

This module handles the conversion and provides flexible loading options.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Tensor operations will be limited.")


# RAF-DB AU indices to standard AU numbers mapping
# RAF uses 26 AUs, we map to standard FACS AU numbers
RAF_AU_MAPPING = {
    0: 1,   # AU1: inner brow raiser
    1: 2,   # AU2: outer brow raiser
    2: 4,   # AU4: brow lowerer
    3: 5,   # AU5: upper lid raiser
    4: 6,   # AU6: cheek raiser
    5: 7,   # AU7: lid tightener
    6: 9,   # AU9: nose wrinkler
    7: 10,  # AU10: upper lip raiser
    8: 11,  # AU11: nasolabial deepener
    9: 12,  # AU12: lip corner puller
    10: 13, # AU13: cheek puffer
    11: 14, # AU14: dimpler
    12: 15, # AU15: lip corner depressor
    13: 16, # AU16: lower lip depressor
    14: 17, # AU17: chin raiser
    15: 18, # AU18: lip puckerer
    16: 20, # AU20: lip stretcher
    17: 22, # AU22: lip funneler
    18: 23, # AU23: lip tightener
    19: 24, # AU24: lip pressor
    20: 25, # AU25: lips part
    21: 26, # AU26: jaw drop
    22: 27, # AU27: mouth stretch
    23: 43, # AU43: eyes closed
    24: 45, # AU45: blink
    25: 46, # AU46: wink
}

# Standard 23 custom AUs used in our framework
CUSTOM_23_AUS = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 18, 20, 23, 24, 25, 26, 27, 28, 43, 45, 46]

# Emotion names in RAF-DB order
RAF_EMOTIONS = [
    'surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger',
    'happiness_surprise', 'happiness_disgust', 'sadness_fear',
    'sadness_anger', 'sadness_surprise', 'sadness_disgust',
    'fear_anger', 'fear_surprise', 'anger_surprise',
    'anger_disgust', 'disgust_surprise'
]

# Basic emotions (training set)
BASIC_EMOTIONS = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger']

# Compound emotions (test set)
COMPOUND_EMOTIONS = [
    'happiness_surprise', 'happiness_disgust', 'sadness_fear',
    'sadness_anger', 'sadness_surprise', 'sadness_disgust',
    'fear_anger', 'fear_surprise', 'anger_surprise',
    'anger_disgust', 'disgust_surprise'
]


class AUEMOPriorLoader:
    """
    Loader for AU-EMO prior knowledge from RAF-DB materials
    """

    def __init__(
        self,
        materials_dir: str = "codes_v251112/materials",
        use_basic_emotions_only: bool = True,
        target_num_aus: int = 23
    ):
        """
        Args:
            materials_dir: Path to materials directory
            use_basic_emotions_only: If True, only load 6 basic emotions
            target_num_aus: Number of AUs in target framework (23 or 26)
        """
        self.materials_dir = Path(materials_dir)
        self.use_basic_emotions_only = use_basic_emotions_only
        self.target_num_aus = target_num_aus

        # Load data
        self._load_raf_graph()
        self._load_au_descriptions()

    def _load_raf_graph(self):
        """Load RAF graph containing ex_au matrix"""
        raf_path = self.materials_dir / "RAF_graph.json"

        if not raf_path.exists():
            raise FileNotFoundError(
                f"RAF_graph.json not found at {raf_path}. "
                f"Please ensure materials are available."
            )

        with open(raf_path, 'r') as f:
            data = json.load(f)

        # Extract ex_au matrix [17, 26]
        self.ex_au_raw = data['ex_au']
        self.emotions_all = data['nodes']
        self.au_au_matrix = data.get('au_au', None)

        # Filter to basic emotions if requested
        if self.use_basic_emotions_only:
            # Find indices of basic emotions
            basic_indices = [
                i for i, emo in enumerate(self.emotions_all)
                if emo in BASIC_EMOTIONS
            ]

            self.emotions = [self.emotions_all[i] for i in basic_indices]
            self.ex_au = [self.ex_au_raw[i] for i in basic_indices]
        else:
            self.emotions = self.emotions_all
            self.ex_au = self.ex_au_raw

        print(f"Loaded AU-EMO prior for {len(self.emotions)} emotions")
        print(f"Emotions: {self.emotions}")

    def _load_au_descriptions(self):
        """Load AU descriptions"""
        au_action_path = self.materials_dir / "AU_action.txt"
        au_desc_path = self.materials_dir / "AU_description.txt"

        self.au_actions = {}
        self.au_descriptions = {}

        # Load AU actions
        if au_action_path.exists():
            with open(au_action_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        au_name, action = line.split(':', 1)
                        self.au_actions[au_name] = action

        # Load AU descriptions
        if au_desc_path.exists():
            with open(au_desc_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        au_name, desc = line.split(':', 1)
                        self.au_descriptions[au_name] = desc

    def get_prior_matrix(
        self,
        emotion_subset: Optional[List[str]] = None,
        normalize: bool = False,
        as_list: bool = False
    ) -> Union[Tuple['torch.Tensor', List[str], List[int]], Tuple[List[List[float]], List[str], List[int]]]:
        """
        Get P(AU | EMO) prior matrix

        Args:
            emotion_subset: Optional list of emotions to include
            normalize: Whether to normalize each emotion's AU distribution
            as_list: Return as list instead of tensor (for when torch unavailable)

        Returns:
            prior_matrix: [num_aus, num_emotions] tensor or list of lists
            emotions: List of emotion names
            au_indices: List of AU indices used
        """
        # Select emotions
        if emotion_subset is not None:
            selected_emotions = emotion_subset
            emotion_indices = [
                self.emotions.index(emo) for emo in emotion_subset
                if emo in self.emotions
            ]
            if len(emotion_indices) != len(emotion_subset):
                missing = set(emotion_subset) - set(self.emotions)
                warnings.warn(f"Some emotions not found: {missing}")
        else:
            selected_emotions = self.emotions
            emotion_indices = list(range(len(self.emotions)))

        # Extract relevant rows
        ex_au_subset = [self.ex_au[i] for i in emotion_indices]

        # Map to target number of AUs
        if self.target_num_aus == 23:
            # Use first 23 AUs (standard subset)
            ex_au_subset = [row[:23] for row in ex_au_subset]
            au_indices = list(range(23))
        elif self.target_num_aus == 26:
            # Use all 26 AUs
            au_indices = list(range(26))
        else:
            raise ValueError(f"Unsupported target_num_aus: {self.target_num_aus}")

        # Transpose to [num_aus, num_emotions]
        num_aus = len(ex_au_subset[0])
        num_emotions = len(ex_au_subset)
        prior_matrix_list = [
            [ex_au_subset[emo_idx][au_idx] for emo_idx in range(num_emotions)]
            for au_idx in range(num_aus)
        ]

        # Normalize if requested
        if normalize:
            # Normalize each emotion's distribution (each column)
            for emo_idx in range(num_emotions):
                col_sum = sum(prior_matrix_list[au_idx][emo_idx] for au_idx in range(num_aus))
                if col_sum > 0:
                    for au_idx in range(num_aus):
                        prior_matrix_list[au_idx][emo_idx] /= col_sum

        # Convert to tensor if torch is available and not requesting list
        if TORCH_AVAILABLE and not as_list:
            prior_matrix = torch.tensor(prior_matrix_list, dtype=torch.float32)
            return prior_matrix, selected_emotions, au_indices
        else:
            return prior_matrix_list, selected_emotions, au_indices

    def get_prior_dict(
        self,
        emotion_subset: Optional[List[str]] = None,
        normalize: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Get P(AU | EMO) prior as nested dictionary

        Returns:
            Dict mapping emotion -> AU -> probability
        """
        prior_matrix, emotions, au_indices = self.get_prior_matrix(
            emotion_subset, normalize, as_list=True
        )

        # Convert to dict
        prior_dict = {}
        for emo_idx, emo in enumerate(emotions):
            prior_dict[emo] = {}
            for au_idx, au in enumerate(au_indices):
                au_name = f"AU{au}" if au < 100 else f"AU{au}"
                # Handle both tensor and list
                if isinstance(prior_matrix, list):
                    prob = prior_matrix[au_idx][emo_idx]
                else:
                    prob = prior_matrix[au_idx, emo_idx].item()
                if prob > 0:  # Only store non-zero probabilities
                    prior_dict[emo][au_name] = prob

        return prior_dict

    def save_prior_json(
        self,
        output_path: str,
        emotion_subset: Optional[List[str]] = None,
        normalize: bool = False
    ):
        """
        Save P(AU | EMO) prior to JSON file

        Args:
            output_path: Path to output JSON file
            emotion_subset: Optional list of emotions to include
            normalize: Whether to normalize
        """
        prior_dict = self.get_prior_dict(emotion_subset, normalize)

        output = {
            'emotions': list(prior_dict.keys()),
            'num_aus': self.target_num_aus,
            'prior': prior_dict,
            'metadata': {
                'source': 'RAF-DB',
                'normalized': normalize,
                'basic_emotions_only': self.use_basic_emotions_only
            }
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Saved AU-EMO prior to {output_path}")

    def get_emotion_stats(self) -> Dict[str, Dict]:
        """
        Get statistics about each emotion's AU activations

        Returns:
            Dict mapping emotion -> stats
        """
        stats = {}

        for emo_idx, emo in enumerate(self.emotions):
            au_probs = self.ex_au[emo_idx]

            # Find active AUs (prob > 0)
            active_aus = [
                (au_idx, prob) for au_idx, prob in enumerate(au_probs)
                if prob > 0
            ]

            # Sort by probability
            active_aus.sort(key=lambda x: x[1], reverse=True)

            stats[emo] = {
                'num_active_aus': len(active_aus),
                'total_activation': sum(prob for _, prob in active_aus),
                'mean_activation': sum(prob for _, prob in active_aus) / len(active_aus) if active_aus else 0,
                'top_aus': [
                    (f"AU{au_idx}", prob) for au_idx, prob in active_aus[:5]
                ]
            }

        return stats

    def print_summary(self):
        """Print summary of loaded AU-EMO prior"""
        print("\n" + "="*80)
        print("AU-EMO Prior Summary")
        print("="*80)

        print(f"\nEmotions: {len(self.emotions)}")
        for i, emo in enumerate(self.emotions):
            print(f"  {i}: {emo}")

        print(f"\nAUs: {self.target_num_aus}")
        print(f"Matrix shape: [{self.target_num_aus}, {len(self.emotions)}]")

        print("\n" + "-"*80)
        print("Emotion Statistics:")
        print("-"*80)

        stats = self.get_emotion_stats()
        for emo, emo_stats in stats.items():
            print(f"\n{emo}:")
            print(f"  Active AUs: {emo_stats['num_active_aus']}")
            print(f"  Total activation: {emo_stats['total_activation']:.2f}")
            print(f"  Top AUs: {emo_stats['top_aus'][:3]}")


def create_prior_for_frameworks(
    materials_dir: str = "codes_v251112/materials",
    output_dir: str = "codes_v251112/continual_learning",
    use_basic_emotions_only: bool = True
):
    """
    Create AU-EMO prior files for all frameworks

    Args:
        materials_dir: Path to materials directory
        output_dir: Output directory for prior files
        use_basic_emotions_only: Whether to use only basic emotions
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prior
    loader = AUEMOPriorLoader(
        materials_dir=materials_dir,
        use_basic_emotions_only=use_basic_emotions_only,
        target_num_aus=23
    )

    # Print summary
    loader.print_summary()

    # Save different versions

    # 1. Basic emotions only (for training)
    loader.save_prior_json(
        output_path=str(output_dir / "au_emo_prior_basic.json"),
        emotion_subset=BASIC_EMOTIONS,
        normalize=False
    )

    # 2. All emotions (for full evaluation)
    loader_all = AUEMOPriorLoader(
        materials_dir=materials_dir,
        use_basic_emotions_only=False,
        target_num_aus=23
    )
    loader_all.save_prior_json(
        output_path=str(output_dir / "au_emo_prior_all.json"),
        emotion_subset=None,
        normalize=False
    )

    # 3. Normalized version (for certain algorithms)
    loader.save_prior_json(
        output_path=str(output_dir / "au_emo_prior_basic_normalized.json"),
        emotion_subset=BASIC_EMOTIONS,
        normalize=True
    )

    print("\n" + "="*80)
    print("Created AU-EMO prior files:")
    print(f"  1. {output_dir / 'au_emo_prior_basic.json'} (6 basic emotions)")
    print(f"  2. {output_dir / 'au_emo_prior_all.json'} (17 all emotions)")
    print(f"  3. {output_dir / 'au_emo_prior_basic_normalized.json'} (normalized)")
    print("="*80)


if __name__ == "__main__":
    # Test the loader
    print("Testing AU-EMO Prior Loader...")

    # Create prior files
    create_prior_for_frameworks(
        materials_dir="codes_v251112/materials",
        output_dir="codes_v251112/continual_learning",
        use_basic_emotions_only=True
    )

    print("\n✓ AU-EMO Prior Loader test completed!")
