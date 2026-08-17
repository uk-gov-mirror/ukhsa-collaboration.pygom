from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


SCHEMA_VERSION = "1.0.0"


class ModelKind(str, Enum):
    DETERMINISTIC_ODE = "deterministic_ode"
    STOCHASTIC = "stochastic"
    HYBRID = "hybrid"


class EventType(str, Enum):
    BETWEEN_STATES = "between_states"
    BIRTH = "birth"
    DEATH = "death"
    EXTERNAL = "external"


class CFMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    title: str
    summary: str | None = None
    institution: str | None = None
    creator_name: str | None = None
    creator_email: str | None = None
    source: str | None = None
    references: list[str] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)
    comment: str | None = None
    conventions: str = "CF-1.13, PyGOM-1.0"
    created: datetime | None = None
    modified: datetime | None = None


class EpiMetadata(BaseModel):
    disease: str | None = None
    pathogen: str | None = None
    pathogen_taxon_id: str | None = None

    geographic_scope: str | None = None
    population_scope: str | None = None

    host_species: str | None = None

    timescale_unit: str = "day"

    model_kind: ModelKind = ModelKind.DETERMINISTIC_ODE

    epidemiological_parameter: list[str] = Field(default_factory=list)

    ontology_references: list[str] = Field(default_factory=list)

    data_sources: list[str] = Field(default_factory=list)

    study_references: list[str] = Field(default_factory=list)


class ScientificVariable(BaseModel):
    name: str
    standard_name: str | None = None
    long_name: str | None = None
    description: str | None = None
    units: str | None = None
    symbol: str | None = None


class StateDefinition(ScientificVariable):
    initial_value: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None


class ParameterDefinition(ScientificVariable):
    value: float | None = None
    expression: str | None = None
    distribution: str | None = None
    citation: str | None = None


class ObservableDefinition(ScientificVariable):
    expression: str


class EventDefinition(BaseModel):
    name: str
    event_type: EventType
    origin: str | None = None
    destination: str | None = None
    rate: str
    description: str | None = None


class CompartmentalModel(BaseModel):
    schema_version: str = SCHEMA_VERSION
    metadata: CFMetadata
    epidemiology: EpiMetadata
    states: list[StateDefinition]
    parameters: list[ParameterDefinition]
    events: list[EventDefinition]
    observables: list[ObservableDefinition] = Field(default_factory=list)
    extensions: dict[str, Any] = Field(default_factory=dict)
