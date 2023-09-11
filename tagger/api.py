"""API module for FastAPI"""
from typing import Callable, Dict, Optional
from threading import Lock
from secrets import compare_digest
import asyncio
from collections import defaultdict

from modules import shared  # pylint: disable=import-error
from modules.api.api import decode_base64_to_image  # pylint: disable=E0401
from modules.call_queue import queue_lock  # pylint: disable=import-error
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from tagger import utils  # pylint: disable=import-error
from tagger import api_models as models  # pylint: disable=import-error


class Api:
    """Api class for FastAPI"""
    def __init__(
        self, app: FastAPI, qlock: Lock, prefix: Optional[str] = None
    ) -> None:
        if shared.cmd_opts.api_auth:
            self.credentials = {}
            for auth in shared.cmd_opts.api_auth.split(","):
                user, password = auth.split(":")
                self.credentials[user] = password

        self.app = app
        self.queue: Dict[str, asyncio.Queue] = {}
        self.results: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.queue_lock = qlock

        self.runner: Optional[asyncio.Task] = None
        self.prefix = prefix

        self.images: Dict[str, Dict[str, Dict[str, tuple[object, float]]]] = \
            defaultdict(lambda: defaultdict(dict))

        self.add_api_route(
            'interrogate',
            self.endpoint_interrogate,
            methods=['POST'],
            response_model=models.TaggerInterrogateResponse
        )

        self.add_api_route(
            'interrogators',
            self.endpoint_interrogators,
            methods=['GET'],
            response_model=models.TaggerInterrogatorsResponse
        )

        self.add_api_route(
            'unload-interrogators',
            self.endpoint_unload_interrogators,
            methods=['POST'],
            response_model=str,
        )

    async def batch_process(self, model: str) -> None:
        done: Dict[str, bool] = {model: False}

        while len(self.queue) > 0:
            while self.queue[model].qsize() > 0:
                with self.queue_lock:
                    q, n, i, t = self.queue[model].get_nowait()
                    if n != "":
                        # Leaving queue and _name empty to process, not queue
                        self.results[q][n] = await self.endpoint_interrogate(
                            models.TaggerInterrogateRequest(
                                image=i,
                                model=model,
                                threshold=t,
                                queue="",
                                queued_name=""
                            )
                        )
                    else:
                        # This is the end of the queue
                        done[model] = True
            if done[model]:
                del self.queue[model]
            await asyncio.sleep(0.1)

    def auth(self, creds: Optional[HTTPBasicCredentials] = None):
        if creds is None:
            creds = Depends(HTTPBasic())
        if creds.username in self.credentials:
            if compare_digest(creds.password,
                              self.credentials[creds.username]):
                return True

        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={
                "WWW-Authenticate": "Basic"
            })

    def add_api_route(self, path: str, endpoint: Callable, **kwargs):
        if self.prefix:
            path = f'{self.prefix}/{path}'

        if shared.cmd_opts.api_auth:
            return self.app.add_api_route(path, endpoint, dependencies=[
                   Depends(self.auth)], **kwargs)
        return self.app.add_api_route(path, endpoint, **kwargs)

    def endpoint_interrogate(self, req: models.TaggerInterrogateRequest):
        """ one file interrogation """
        if req.image is None:
            raise HTTPException(404, 'Image not found')

        if req.model not in utils.interrogators:
            raise HTTPException(404, 'Model not found')

        image = decode_base64_to_image(req.image)
        res: Dict[str, Dict[str, float]] = defaultdict(dict)
        m, q, n = (req.model, req.queue, req.name)

        with self.queue_lock:
            if q != '':
                # clobber existing image
                if n in self.images[m][q]:
                    i = 0
                    while f'{n}.{i}' in self.images[m][q]:
                        i = i + 1
                    n = f'{n}.{i}'

                if m in self.queue:
                    # add image to queue
                    self.queue[m].put_nowait((q, n, image, req.threshold))
                else:
                    self.queue[m] = asyncio.Queue()
                    self.queue[m].put_nowait((q, n, image, req.threshold))
                    self.runner = asyncio.create_task(self.batch_process(m))
                if n == '':
                    res = self.results[q]
            else:
                interrogator = utils.interrogators[m]
                rating, tag = interrogator.interrogate(image)

                res[n] = {**rating}
                for k, v in tag.items():
                    if v > req.threshold:
                        res[n][k] = v

        return models.TaggerInterrogateResponse(caption=res)

    def endpoint_interrogators(self):
        return models.TaggerInterrogatorsResponse(
            models=list(utils.interrogators.keys())
        )

    def endpoint_unload_interrogators(self):
        unloaded_models = 0

        for i in utils.interrogators.values():
            if i.unload():
                unloaded_models = unloaded_models + 1

        return f"Successfully unload {unloaded_models} model(s)"


def on_app_started(_, app: FastAPI):
    Api(app, queue_lock, '/tagger/v1')
