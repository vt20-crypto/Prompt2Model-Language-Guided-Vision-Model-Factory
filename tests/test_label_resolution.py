from prompt2model.config import RequestedLabel
from prompt2model.label_resolution import LabelResolver


def test_lexical_label_resolution_matches_synonyms() -> None:
    resolver = LabelResolver(enable_clip=False)
    resolved = resolver.resolve(
        requested_labels=[RequestedLabel(name="motorcycle", synonyms=["two wheeler"])],
        dataset_labels=["cat", "dog", "motorbike"],
    )
    assert resolved[0].dataset_label == "motorbike"
    assert resolved[0].method == "lexical"

