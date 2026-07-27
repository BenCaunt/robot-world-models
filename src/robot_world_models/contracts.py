from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
VISUAL_MLP_IMPLEMENTATION = (
    "robot_world_models.models.visual_latent:VisualLatentDynamics"
)
VISUAL_TRANSFORMER_IMPLEMENTATION = (
    "robot_world_models.models.visual_transformer:VisualSpatiotemporalTransformer"
)


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ManifestBase(ContractModel):
    schema_version: Literal[1]
    kind: str
    id: str
    display_name: str
    known_limitations: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not ID_PATTERN.fullmatch(value):
            raise ValueError("id must be a lowercase kebab-case identifier")
        return value


class WarmHubRecord(ContractModel):
    repo: str
    wref: str
    observed_pinned_wref: str
    attribution_url: str

    @field_validator("repo")
    @classmethod
    def validate_repo(cls, value: str) -> str:
        if value.count("/") != 1:
            raise ValueError("repo must be org/repo")
        return value

    @model_validator(mode="after")
    def validate_wrefs(self) -> WarmHubRecord:
        if "@v" in self.wref:
            raise ValueError("wref must be durable and unpinned")
        if "@v" not in self.observed_pinned_wref:
            raise ValueError("observed_pinned_wref must record an observed version")
        if self.observed_pinned_wref.split("@v", maxsplit=1)[0] != self.wref:
            raise ValueError("observed_pinned_wref must pin wref")
        return self


class AdapterRef(ContractModel):
    adapter: str
    implementation_status: Literal["implemented", "planned"]
    location_resolved_from_warmhub: bool


class FormatRef(ContractModel):
    adapter: str
    implementation_status: Literal["implemented", "planned"]
    codebase_version: str


class RobotEvidence(ContractModel):
    recorded_with_wref: str
    robot_wref: str
    match_method: str
    confidence: float = Field(ge=0, le=1)
    notes: str
    requires_user_confirmation: bool


class VectorFeature(ContractModel):
    dimension: int = Field(gt=0)
    names: list[str]
    units: str

    @model_validator(mode="after")
    def validate_dimension(self) -> VectorFeature:
        if len(self.names) != self.dimension:
            raise ValueError("dimension must equal the number of feature names")
        return self


class EpisodeSchema(ContractModel):
    fps: float = Field(gt=0)
    total_episodes: int = Field(gt=0)
    total_frames: int = Field(gt=0)
    state: VectorFeature
    action: VectorFeature
    cameras: list[str]


class DatasetMixtureLabels(ContractModel):
    embodiment: str
    schema_family: str
    domain: str
    compatibility_tier: str


class NestedDatasetCollection(ContractModel):
    layout: Literal["nested-lerobot-v3"]
    members: list[str] = Field(min_length=1)
    excluded_members: list[str] = Field(default_factory=list)
    selection_evidence: str

    @model_validator(mode="after")
    def validate_members(self) -> NestedDatasetCollection:
        if len(set(self.members)) != len(self.members):
            raise ValueError("collection members must be unique")
        if len(set(self.excluded_members)) != len(self.excluded_members):
            raise ValueError("excluded collection members must be unique")
        overlap = set(self.members) & set(self.excluded_members)
        if overlap:
            raise ValueError(f"collection members cannot also be excluded: {sorted(overlap)}")
        unsafe = [
            member
            for member in [*self.members, *self.excluded_members]
            if member.startswith("/")
            or not member
            or any(part in {"", ".", ".."} for part in member.split("/"))
        ]
        if unsafe:
            raise ValueError(f"collection member roots must be safe relative paths: {unsafe}")
        return self


class DatasetStoragePolicy(ContractModel):
    upstream_bytes: int = Field(gt=0)
    warmhub_payload_bytes: Literal[0]
    policy: Literal["metadata-in-warmhub-payload-upstream"]


class DatasetManifest(ManifestBase):
    kind: Literal["dataset"]
    warmhub: WarmHubRecord
    profile_wref: str
    license: str
    upstream_revision: str
    source: AdapterRef
    format: FormatRef
    robot_evidence: RobotEvidence
    modalities: list[str]
    episode_schema: EpisodeSchema
    mixture: DatasetMixtureLabels
    required_assessments: list[str]
    fixture: str | None
    collection: NestedDatasetCollection | None = None
    storage: DatasetStoragePolicy | None = None


class RobotDescription(ContractModel):
    wref: str
    observed_pinned_wref: str
    model_profile_wref: str
    format: Literal["urdf", "mjcf"]
    license: str
    redistributable: bool
    pinned_commit: str
    source_repo: str
    package_root: str
    entrypoint: str


class RobotModelProfile(ContractModel):
    dof: int = Field(gt=0)
    joint_count: int = Field(gt=0)
    link_count: int = Field(gt=0)
    mesh_count: int = Field(ge=0)
    mass_total_kg: float = Field(ge=0)
    has_collision_geometry: bool


class JointTransform(ContractModel):
    urdf_joint: str
    scale: float
    offset: float
    evidence: list[str] = Field(min_length=1)


class JointMapping(ContractModel):
    status: Literal["provisional", "validated"]
    coverage: Literal["partial", "complete"]
    animate_in_rerun: bool
    dataset_units: str
    urdf_units: str
    out_of_range_policy: Literal["clamp", "reject"]
    entries: dict[str, JointTransform] = Field(min_length=1)
    unmapped_features: list[str] = Field(default_factory=list)
    validation_required: list[str]

    @model_validator(mode="after")
    def prevent_unvalidated_animation(self) -> JointMapping:
        if self.animate_in_rerun and self.status != "validated":
            raise ValueError("Rerun animation requires a validated joint mapping")
        if self.coverage == "complete" and self.unmapped_features:
            raise ValueError("complete mappings cannot declare unmapped features")
        if self.coverage == "partial" and not self.unmapped_features:
            raise ValueError("partial mappings must declare unmapped features")
        if set(self.entries) & set(self.unmapped_features):
            raise ValueError("a feature cannot be both mapped and unmapped")
        urdf_joints = [entry.urdf_joint for entry in self.entries.values()]
        if len(set(urdf_joints)) != len(urdf_joints):
            raise ValueError("each mapped feature must target a distinct URDF joint")
        return self


class RobotManifest(ManifestBase):
    kind: Literal["robot"]
    warmhub: WarmHubRecord
    description: RobotDescription
    source: AdapterRef
    model_profile: RobotModelProfile
    required_assessments: list[str]


class JointMappingManifest(ManifestBase):
    kind: Literal["joint-mapping"]
    dataset: str
    robot: str
    mapping: JointMapping


class RecipeIntent(ContractModel):
    prediction_target: str
    horizon_steps: int = Field(gt=0)
    purpose: str


class SplitPolicy(ContractModel):
    unit: Literal["episode", "source", "time"]
    train_fraction: float = Field(gt=0, lt=1)
    validation_fraction: float = Field(gt=0, lt=1)
    test_fraction: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def validate_total(self) -> SplitPolicy:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1")
        return self


class RecipeMixture(ContractModel):
    type: Literal["homogeneous", "heterogeneous"]
    datasets: list[str] = Field(min_length=1)
    robot: str
    sampling: str
    missing_modality_policy: str
    normalization: str
    split: SplitPolicy


class RecipeModalities(ContractModel):
    required: list[str] = Field(min_length=1)
    optional: list[str] = Field(default_factory=list)


class VisionEncoderContract(ContractModel):
    model_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_url: str
    license: str
    warmhub_resolution: Literal["resolved", "registry-gap"]
    frozen: Literal[True]
    input_size: int = Field(gt=0)
    patch_pool_grid: int = Field(gt=0)
    latent_dimension: int = Field(gt=0)


class VisionModelContract(ContractModel):
    camera: str
    context_frames: int = Field(gt=0)
    training_rollout_horizon: int = Field(gt=0)
    rollout_loss_discount: float = Field(gt=0, le=1)
    output_size: int = Field(gt=0)
    predictor_hidden_dimension: int = Field(gt=0)
    predictor_hidden_layers: int = Field(gt=0)
    attention_heads: int | None = Field(default=None, gt=0)
    encoder_batch_size: int = Field(gt=0)
    state_loss_weight: float = Field(ge=0)
    decoder_loss_weight: float = Field(ge=0)
    predicted_pixel_loss_weight: float = Field(ge=0)
    encoder: VisionEncoderContract


class ModelContract(ContractModel):
    family: str
    implementation: str
    state_dimension: int = Field(gt=0)
    action_dimension: int = Field(gt=0)
    hidden_dimension: int = Field(gt=0)
    hidden_layers: int = Field(gt=0)
    vision: VisionModelContract | None = None


class TrainingSubset(ContractModel):
    max_episodes: int = Field(gt=0)
    member_roots: list[str] = Field(default_factory=list)
    max_download_bytes: int | None = Field(default=None, gt=0)

    @field_validator("member_roots")
    @classmethod
    def validate_member_roots(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("training subset member_roots must be unique")
        return value


class TrainingContract(ContractModel):
    seed: int
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    max_steps: int = Field(gt=0)
    smoke_test_steps: int = Field(gt=0)
    checkpoint_every_steps: int = Field(gt=0)
    local_devices: list[Literal["mps", "cuda", "cpu"]] = Field(min_length=1)
    subset: TrainingSubset


class EvaluationContract(ContractModel):
    rerun_required: Literal[True]
    urdf_required: bool
    urdf_animation_requires_validated_joint_mapping: bool
    metrics: list[str] = Field(min_length=1)
    rollout_horizons: list[int] = Field(min_length=1)


class RemoteComputeContract(ContractModel):
    enabled: bool
    implementation_status: Literal["plan-only", "implemented"]
    provider: Literal["runpod"]
    preferred_gpu: str
    gpu_count: Literal[1]
    cloud_type: Literal["secure", "community"]
    max_hourly_usd: float | None = Field(default=None, gt=0)
    max_runtime_minutes: int | None = Field(default=None, gt=0)
    persistent_volume: bool
    public_service_ports: bool
    explicit_approval_required: Literal[True]


class RecipeManifest(ManifestBase):
    kind: Literal["recipe"]
    intent: RecipeIntent
    mixture: RecipeMixture
    joint_mapping: str | None = None
    modalities: RecipeModalities
    model: ModelContract
    training: TrainingContract
    evaluation: EvaluationContract
    remote_compute: RemoteComputeContract

    @model_validator(mode="after")
    def validate_visual_training_horizon(self) -> RecipeManifest:
        if (
            self.model.vision is not None
            and self.intent.horizon_steps
            != self.model.vision.training_rollout_horizon
        ):
            raise ValueError(
                "visual intent horizon_steps must equal training_rollout_horizon"
            )
        return self

    @model_validator(mode="after")
    def validate_visual_attention(self) -> RecipeManifest:
        vision = self.model.vision
        if vision is None:
            return self
        if (
            self.model.implementation == VISUAL_TRANSFORMER_IMPLEMENTATION
            and vision.attention_heads is None
        ):
            raise ValueError("visual transformer requires attention_heads")
        if (
            self.model.implementation == VISUAL_TRANSFORMER_IMPLEMENTATION
            and vision.predictor_hidden_layers < 2
        ):
            raise ValueError("visual transformer requires at least two predictor_hidden_layers")
        if vision.attention_heads is None:
            return self
        if vision.predictor_hidden_dimension % vision.attention_heads:
            raise ValueError(
                "predictor_hidden_dimension must be divisible by attention_heads"
            )
        return self


Manifest = Annotated[
    DatasetManifest | RobotManifest | JointMappingManifest | RecipeManifest,
    Field(discriminator="kind"),
]
MANIFEST_ADAPTER = TypeAdapter(Manifest)
