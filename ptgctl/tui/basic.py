from rich.panel import Panel
from textual.app import App
from textual.widget import Widget
from textual.widgets import Placeholder

import ptgctl
import ptgctl.holoframe
from ptgctl.tools.display import ascii_image



class Image(Widget):
    def __init__(self, api, stream_id, *a, **kw):
        self.api = api
        self.stream_id = stream_id
        super().__init__(*a, **kw)

    async def on_mount(self):
        self.last = 0
        self.data = None
        self.ws = await self.api.data_pull_connect(self.stream_id, last_entry_id=0).__aenter__()
        self.set_interval(0.05, self.refresh_image)

    async def on_unmount(self):
        await self.ws.__aexit__()
        await super().on_unmount()

    async def refresh_image(self, *a, **kw):
        data = await self.ws.recv_data()
        self.data = ptgctl.holoframe.load_all(data)[self.stream_id]
        self.refresh(*a, **kw)

    def render(self):
        return Panel(
            ascii_image(self.data['image'], self._size[0], self._size[1])
            if self.data else '')


class ImageApp(App):
    def __init__(self, api, stream_ids, *a, **kw):
        self.api = api
        self.stream_ids = stream_ids
        super().__init__(*a, **kw)

    async def on_mount(self):
        grid = await self.view.dock_grid()

        grid.add_column("col", fraction=1, max_size=120)
        grid.add_row("row", fraction=1, max_size=60)
        grid.set_repeat(True, True)
        grid.add_areas(center="col-2-start|col-4-end,row-2-start|row-3-end")
        grid.set_align("stretch", "center")
        grid.place(*[Image(self.api, sid) for sid in self.stream_ids])



def main(*stream_ids, api=None):
    api = api or ptgctl.API(local=False)
    ImageApp.run(api=api, stream_ids=stream_ids or ['main'])

if __name__ == '__main__':
    import fire
    fire.Fire(main)