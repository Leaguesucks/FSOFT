from API.api import API

from Tools.CodeExecutor import Executor

api = API()
app = api.app

executor = Executor()
app.include_router(executor.router)