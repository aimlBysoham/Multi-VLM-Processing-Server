# basic imports
import queue
import base64
import logging
import numpy as np
from io import BytesIO

# fast-api imports
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel