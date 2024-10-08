from src.router             import Router 
from src.processing_element import ProcessingElement

class Simulator: 
    def __init__(self, num_rows:int, num_cols:int, debug_mode: bool = False):
        self._debug_mode    = debug_mode   
        self._num_rows      = num_rows
        self._num_cols      = num_cols

        self._routers       = self._create_routers()
        self._pes           = self._create_pes()

    def _create_routers(self) -> dict[tuple[int, int], Router]:
        router_lookup = {}
        for x in range(self._num_rows):
            for y in range(self._num_cols):
                router = Router( pos=(x, y), debug_mode=self._debug_mode )
                router_lookup[(x, y)] = router
        return router_lookup

    def _create_pes(self) -> dict[tuple[int, int], ProcessingElement]:
        pe_lookup = {}
        for x in range(self._num_rows):
            for y in range(self._num_cols):
                pe = ProcessingElement( xy=(x, y), debug_mode=self._debug_mode, router_lookup=self._routers )
                pe_lookup[(x, y)] = pe
        return pe_lookup

    def run(self) -> None:
        print(f"\nRunning simulation with {self._num_rows}x{self._num_cols} mesh PEs")

        print(f"\nInitializing routers and PEs\n")
        for router, pe in zip(self._routers.values(), self._pes.values()):
            print(f"Router: {router} -> PE: {pe}")


if __name__ == "__main__":
    sim = Simulator(4, 4, 16)
    sim.run()