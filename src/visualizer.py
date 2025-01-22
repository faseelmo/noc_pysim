
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

from .router import Router
from .flit import EmptyFlit, HeaderFlit, PayloadFlit, TailFlit
from .processing_element import ProcessingElement
from .simulator import Map


class Visualizer:
    def __init__(self, num_rows: int, num_cols:int,  routers: list[Router], pes: list[ProcessingElement]) -> None: 
        self._router            = routers 
        self._pes               = pes
        self._mapping_list      = []  

        self._color_map         = { }
        self._num_rows          = num_rows
        self._num_cols          = num_cols

        # Matplotlib parameters
        self._buffer_spacing    = 1.4
        self._label_font_size   = 8
        self._flit_font_size    = 14
        self._label_offset      = 0.3
        self._flit_offset       = 0.3
        self._lim_spacing       = 2.4

        self._next_cycle        = False # Flag to move to next cycle

        self.fig, self.axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))

        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        plt.ion()

    def init_mapping(self, mapping_list: list[Map]) -> None:
        self._mapping_list = mapping_list

    def __call__(self, cycle_count: int) -> None:

        self._next_cycle = False

        for router in self._router.values():
            router_xy = router.get_pos()
            x, y = self._transform_xy(router_xy[0], router_xy[1])

            ax = self.axes[x, y]
            self._draw_router(router, ax)

        for pe in self._pes.values():
            pe_xy = pe.get_pos()
            x, y = self._transform_xy(pe_xy[0], pe_xy[1])

            ax = self.axes[x, y]
            self._draw_pe(pe, ax)

        plt.suptitle(f"Cycle: {cycle_count}")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)  # Remove spacing
        plt.draw()
            
        while not self._next_cycle:
            plt.waitforbuttonpress()

    def _on_key_press(self, event) -> None:
        if event.key == 'enter':
            self._next_cycle = True

    def _draw_pe(self, pe: ProcessingElement, ax) -> None:

        assigned_task = ""

        for map in self._mapping_list:
            if pe.get_pos() == map.assigned_pe:
                assigned_task = map.task.task_id    

        box_size = 0.5
        if pe.get_status():
            fill_box = Rectangle((-box_size, -box_size), box_size*2, box_size*2, linewidth=0, edgecolor='none', facecolor='red', alpha=0.2)
            ax.add_patch(fill_box)

        ax.text(0, 0, f"Task: {assigned_task}", ha='center', va='center', fontsize=12, color='black')
        ax.axis('off')

    def _draw_router(self, router: Router, ax) -> None:

        buffer_layout       = {
                                "Local" : (0, 0),
                                "West"  : (-self._buffer_spacing, 0),
                                "North" : (0, self._buffer_spacing),
                                "East"  : (self._buffer_spacing, 0),
                                "South" : (0, -self._buffer_spacing)
                            }
        
        buffer_directions   = {
                                "local_input"   : router._local_input_buffer,
                                "local_output"  : router._local_output_buffer,
                                "west_input"    : router._west_input_buffer,
                                "west_output"   : router._west_output_buffer,
                                "north_input"   : router._north_input_buffer,
                                "north_output"  : router._north_output_buffer,
                                "east_input"    : router._east_input_buffer,
                                "east_output"   : router._east_output_buffer,
                                "south_input"   : router._south_input_buffer,
                                "south_output"  : router._south_output_buffer,
                              }
        
        ax.clear() # Clear the axes before drawing the new router state
        
        for buffer_name, buffer_obj in buffer_directions.items():

            direction = buffer_name.split('_')[0].capitalize()
            buf_type = buffer_name.split('_')[1].capitalize()
            position = buffer_layout[direction]
    
            if buf_type == "Input":
                position = (position[0], position[1] - self._label_offset)
            elif buf_type == "Output": 
                position = (position[0], position[1] + self._label_offset)
    
            label_position = (position[0], position[1]) 
    
            ax.text( # Draw the buffer label box
                label_position[0], 
                label_position[1], 
                f"{direction} {buf_type}", 
                ha='center', 
                va='center', 
                fontsize=self._label_font_size, 
                color='black', 
                bbox=dict(facecolor='white', edgecolor='black'))
    
            for idx, flit in enumerate(buffer_obj.queue):
                # Drawing flits
                if isinstance(flit, EmptyFlit):
                    continue
                
                if isinstance(flit, HeaderFlit):
                    label = 'H'
                elif isinstance(flit, PayloadFlit):
                    label = 'P'
                elif isinstance(flit, TailFlit):
                    label = 'T'
                else:
                    label = '?'
    
                packet_uid = flit.get_uid()
                if packet_uid not in self._color_map:
                    # Assigning unique color to each packet
                    self._color_map[packet_uid] = ( random.random(), random.random(), random.random() )
                color = self._color_map[packet_uid]
    
                if buf_type == "Input":
                    flit_y_offset = -self._flit_offset
                elif buf_type == "Output":
                    flit_y_offset = self._flit_offset
    
                ax.add_patch(
                    plt.Circle(
                        (position[0] + 0.3 * idx, position[1] + flit_y_offset), 
                        0.15, color=color, 
                        ec='black') )

                ax.text(
                    position[0] + 0.3 * idx, position[1] + flit_y_offset, 
                    label, 
                    fontsize=self._flit_font_size, 
                    ha='center', 
                    va='center', 
                    color='white')
        
        ax.set_xlim(-self._lim_spacing, self._lim_spacing)
        ax.set_ylim(-self._lim_spacing, self._lim_spacing)
    
        ax.set_aspect('equal')

        ax.text(
            -self._lim_spacing, 
            self._lim_spacing - 0.4,
            f"R({router._x}, {router._y})", 
            ha='left', 
            va='top', 
            fontsize=12, 
            color='black')
        
        ax.axis('off')


    def _transform_xy(self, x:int, y:int) -> tuple[int, int]:
        """
        Converts router (x,y) to matplotlib (x,y)

        Router (x,y) origin is at the bottom left 
        Matplotlib origin is at the top left
        """
        transformed_x = self._num_rows - 1 - y
        transformed_y = x
        return transformed_x, transformed_y


    