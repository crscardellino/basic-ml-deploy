"""
This is the main application that loads and serves a FastText model via a REST
API using FastAPI.

    MIT License
    Copyright (c) 2024 Cristian Cardellino.
"""

import logging
import os
import fasttext
import time

from contextlib import asynccontextmanager
from dataclasses import dataclass
from fastapi import APIRouter, HTTPException, FastAPI
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class GlobalConfig:
    """
    Class to handle the configuration globally.
    The class is instantiated during the startup event of a FastAPI app.
    It is a Python dataclass.

    Attributes
    ----------
        config: DictConfig
            An OmegaConf's DictConfig that is loaded from a YAML file.
            Holds a list of key -> value pairs with the configuration
            for the API.
        model: fasttext.FastText._FastText | None
            The FastText model.
    """

    config: DictConfig = OmegaConf.create()
    model: fasttext.FastText._FastText | None = None


config = GlobalConfig()


class WordVectorArgs(BaseModel):
    """
    Arguments for the FastText's API call.
    It is based on a Pydantic's BaseModel.

    Attributes
    ----------
        text: str
            The text to be transformed by the FastText model.
    """

    text: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "deploying with fastapi",
                }
            ]
        }
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan function that runs as startup and shutdown function for FastAPI.
    """
    config_path = os.getenv("FASTTEXT_ML_CONFIG", "./config/config.yaml")
    logger.info(f"Loading configuration for FastText App from {config_path}.")
    config.config = OmegaConf.load(config_path)
    logger.info("Loading models for FastText App.")
    start = time.time()
    config.model = fasttext.load_model(config.config.model_location)
    end = time.time()
    logger.info(f"Models loaded, time elapsed: {end-start}")
    yield
    # The part after the yield is executed before shutdown, if we need to do
    # some memory cleaning or resource liberation


app = FastAPI(
    title="FastText API",
    description="FastText API to generate word and sentence embeddings",
    version="0.1",
    lifespan=lifespan,
    openapi_url="/api/fasttext/docs/openapi.json",
    docs_url="/api/fasttext/docs/",
)
router = APIRouter(prefix="/api/fasttext")


@router.get("/health")
def health() -> str:
    """
    Let's Docker know if the container it's working.

    Returns
    -------
    str
        The "ok" string.
    """
    return "ok"


@router.post("/word_vectors")
def word_vector(args: WordVectorArgs) -> dict[str, list[float]]:
    """
    Returns the vector for each word in the text.

    Parameters
    ----------
        args: WordVectorArgs
            The args for the POST request (JSON) as part of the payload.

    Returns
    -------
        dict[str, list[float]]
            A mapping between each unique word in the text and its
            corresponding vector.

    Raises
    ------
        HTTPException
            A 400 error if there are no words.
    """
    # Check the input isn't empty
    if len(args.text.strip()) == 0:
        error_message = "The request is empty"
        logger.info(error_message)
        raise HTTPException(status_code=400, detail=error_message)

    output = {}
    for word in sorted(set(args.text.strip().split())):
        word_vector = config.model.get_word_vector(word).tolist()
        if config.config.model_precision is not None:
            word_vector = [
                round(d, config.config.model_precision) for d in word_vector
            ]
        output[word] = word_vector

    return output


app.include_router(router)
