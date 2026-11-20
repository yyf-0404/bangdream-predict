"""Inference from the published conditional curve and timestamped CN cutoff records."""
from data import raw
from data.curve_history import as_of_input, read_observations
from data.joint_sequences import build_joint_event, event_calendar, validate_joint_event
from data.joint_slicing import joint_model_input
from model.conditional_curve import ConditionalCurveModel, DEFAULT_MODEL_PATH
from utils.files import file_sha256


class ConditionalPredictor:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model = ConditionalCurveModel.load(model_path)
        self.model_hash = file_sha256(model_path)
        self.reward_table = self.model.reward_table

    def calendar(self, event_id, *, server=3, metadata=None, characters=None):
        return event_calendar(event_id, server, raw.get_event_data() if metadata is None else metadata,
                              raw.get_character_data() if characters is None else characters, self.reward_table)

    def build_event(self, event_id, **options):
        if 'reward_table' in options:
            raise ValueError('Inference uses the reward table embedded in this model')
        return build_joint_event(event_id, reward_table=self.reward_table, **options)

    def _predict(self, x, cut):
        raw_curve, published = self.model.predict(x)
        return dict(t0=int(cut), ranks=x['ranks'], anchor=x['anchor'], anchor_mask=x['anchor_mask'],
                    raw_cutoff=raw_curve, pred_cutoff=published, model_sha256=self.model_hash)

    def predict_event(self, event, t0, *, requested_rank=None):
        validate_joint_event(event)
        if event['reward_schedule_sha256'] != self.reward_table.sha256:
            raise ValueError('Event/model reward table mismatch')
        return self._predict(joint_model_input(event, t0, requested_rank=requested_rank), t0)

    def predict_at(self, event, series, issued_at_ms, *, ranks=None):
        inputs = as_of_input(event, series, issued_at_ms, ranks=ranks)
        if inputs is None:
            return None
        x, cut = inputs
        return dict(self._predict(x, cut), issued_at_ms=int(issued_at_ms), anchor_at_ms=int(event['grid_ms'][cut]))

    def predict_as_of(self, event_id, issued_at_ms, *, raw_dir=None, metadata=None, characters=None, server=3, ranks=None):
        event = self.calendar(event_id, server=server, metadata=metadata, characters=characters)
        series = read_observations(raw.config.raw_dir if raw_dir is None else raw_dir, event, require_complete=False)
        result = self.predict_at(event, series, issued_at_ms, ranks=ranks)
        if result is None:
            raise ValueError('A pre-settlement issue needs 24 hours of jointly observed history')
        return result
