import tkinter as tk
from tkinter import ttk, font
import threading
import time
import queue
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class MarketData:
    """Container for market data"""
    bids: np.ndarray
    asks: np.ndarray
    mid_price: float
    position: float
    cash: float
    pnl: float
    passive_bid_order: Optional[Tuple[float, float]] = None  # (price, size)
    passive_ask_order: Optional[Tuple[float, float]] = None  # (price, size)
    step: int = 0
    last_action: str = "DO_NOTHING"

class DOMVisualizerGUI:
    """Professional DOM Ladder Visualization"""
    
    def __init__(self, levels: int = 20):
        self.levels = levels
        self.root = tk.Tk()
        self.root.title("LOB Trading Agent - DOM Visualizer")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1e1e1e')
        
        # Data queue for thread-safe updates
        self.data_queue = queue.Queue()
        self.running = True
        
        # Colors
        self.colors = {
            'bg': '#1e1e1e',
            'panel': '#2d2d2d',
            'bid': '#0d4f3c',
            'ask': '#4a1e1e',
            'text': '#ffffff',
            'highlight': '#ffd700',
            'passive_order': '#ff6b35',
            'mid': '#666666'
        }
        
        # Fonts
        self.fonts = {
            'title': ('Consolas', 14, 'bold'),
            'header': ('Consolas', 10, 'bold'),
            'data': ('Consolas', 9),
            'position': ('Consolas', 12, 'bold')
        }
        
        self.setup_gui()
        self.start_update_thread()
    
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(
            main_frame, 
            text="LIMIT ORDER BOOK - AGENT VISUALIZATION",
            font=self.fonts['title'],
            bg=self.colors['bg'],
            fg=self.colors['text']
        )
        title_label.pack(pady=(0, 20))
        
        # Top panel - Position and metrics
        self.setup_position_panel(main_frame)
        
        # Main trading panel
        trading_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        trading_frame.pack(fill=tk.BOTH, expand=True)
        
        # DOM Ladder
        self.setup_dom_ladder(trading_frame)
        
        # Side panel - Agent info
        self.setup_agent_panel(trading_frame)
    
    def setup_position_panel(self, parent):
        """Setup position and P&L display"""
        position_frame = tk.Frame(parent, bg=self.colors['panel'], relief=tk.RAISED, bd=2)
        position_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Position info
        self.position_label = tk.Label(
            position_frame,
            text="POSITION: 0.0",
            font=self.fonts['position'],
            bg=self.colors['panel'],
            fg=self.colors['text']
        )
        self.position_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.pnl_label = tk.Label(
            position_frame,
            text="P&L: $0.00",
            font=self.fonts['position'],
            bg=self.colors['panel'],
            fg=self.colors['text']
        )
        self.pnl_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.cash_label = tk.Label(
            position_frame,
            text="CASH: $0.00",
            font=self.fonts['position'],
            bg=self.colors['panel'],
            fg=self.colors['text']
        )
        self.cash_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.step_label = tk.Label(
            position_frame,
            text="STEP: 0",
            font=self.fonts['position'],
            bg=self.colors['panel'],
            fg=self.colors['text']
        )
        self.step_label.pack(side=tk.RIGHT, padx=20, pady=10)
        
        self.action_label = tk.Label(
            position_frame,
            text="LAST ACTION: DO_NOTHING",
            font=self.fonts['position'],
            bg=self.colors['panel'],
            fg=self.colors['text']
        )
        self.action_label.pack(side=tk.RIGHT, padx=20, pady=10)
    
    def setup_dom_ladder(self, parent):
        """Setup the DOM ladder display"""
        dom_frame = tk.Frame(parent, bg=self.colors['bg'])
        dom_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # DOM Header
        header_frame = tk.Frame(dom_frame, bg=self.colors['panel'], relief=tk.RAISED, bd=1)
        header_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Column headers
        headers = ['BID VOL', 'BID SIZE', 'BID PRICE', 'ASK PRICE', 'ASK SIZE', 'ASK VOL', 'ORDERS']
        header_widths = [10, 10, 12, 12, 10, 10, 15]
        
        for i, (header, width) in enumerate(zip(headers, header_widths)):
            label = tk.Label(
                header_frame,
                text=header,
                font=self.fonts['header'],
                bg=self.colors['panel'],
                fg=self.colors['text'],
                width=width
            )
            label.grid(row=0, column=i, padx=1, pady=5)
        
        # Scrollable DOM area
        dom_canvas = tk.Canvas(dom_frame, bg=self.colors['bg'])
        dom_scrollbar = ttk.Scrollbar(dom_frame, orient="vertical", command=dom_canvas.yview)
        self.dom_content_frame = tk.Frame(dom_canvas, bg=self.colors['bg'])
        
        self.dom_content_frame.bind(
            "<Configure>",
            lambda e: dom_canvas.configure(scrollregion=dom_canvas.bbox("all"))
        )
        
        dom_canvas.create_window((0, 0), window=self.dom_content_frame, anchor="nw")
        dom_canvas.configure(yscrollcommand=dom_scrollbar.set)
        
        dom_canvas.pack(side="left", fill="both", expand=True)
        dom_scrollbar.pack(side="right", fill="y")
        
        # Initialize DOM rows
        self.dom_rows = []
        for i in range(self.levels):
            row_frame = tk.Frame(self.dom_content_frame, bg=self.colors['bg'])
            row_frame.pack(fill=tk.X, pady=1)
            
            # Create labels for each column
            row_labels = []
            for j, width in enumerate(header_widths):
                bg_color = self.colors['bid'] if j < 3 else self.colors['ask'] if j < 6 else self.colors['panel']
                
                label = tk.Label(
                    row_frame,
                    text="",
                    font=self.fonts['data'],
                    bg=bg_color,
                    fg=self.colors['text'],
                    width=width,
                    relief=tk.RAISED,
                    bd=1
                )
                label.grid(row=0, column=j, padx=1)
                row_labels.append(label)
            
            self.dom_rows.append(row_labels)
    
    def setup_agent_panel(self, parent):
        """Setup agent information panel"""
        agent_frame = tk.Frame(parent, bg=self.colors['panel'], relief=tk.RAISED, bd=2)
        agent_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        
        # Agent title
        agent_title = tk.Label(
            agent_frame,
            text="AGENT STATUS",
            font=self.fonts['title'],
            bg=self.colors['panel'],
            fg=self.colors['text']
        )
        agent_title.pack(pady=20)
        
        # Working orders section
        orders_frame = tk.LabelFrame(
            agent_frame,
            text="Working Orders",
            font=self.fonts['header'],
            bg=self.colors['panel'],
            fg=self.colors['text'],
            bd=2
        )
        orders_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.bid_order_label = tk.Label(
            orders_frame,
            text="BID: None",
            font=self.fonts['data'],
            bg=self.colors['panel'],
            fg=self.colors['text'],
            anchor='w'
        )
        self.bid_order_label.pack(fill=tk.X, padx=10, pady=5)
        
        self.ask_order_label = tk.Label(
            orders_frame,
            text="ASK: None",
            font=self.fonts['data'],
            bg=self.colors['panel'],
            fg=self.colors['text'],
            anchor='w'
        )
        self.ask_order_label.pack(fill=tk.X, padx=10, pady=5)
        
        # Market info section
        market_frame = tk.LabelFrame(
            agent_frame,
            text="Market Info",
            font=self.fonts['header'],
            bg=self.colors['panel'],
            fg=self.colors['text'],
            bd=2
        )
        market_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.mid_price_label = tk.Label(
            market_frame,
            text="MID: $0.00",
            font=self.fonts['data'],
            bg=self.colors['panel'],
            fg=self.colors['text'],
            anchor='w'
        )
        self.mid_price_label.pack(fill=tk.X, padx=10, pady=5)
        
        self.spread_label = tk.Label(
            market_frame,
            text="SPREAD: $0.00",
            font=self.fonts['data'],
            bg=self.colors['panel'],
            fg=self.colors['text'],
            anchor='w'
        )
        self.spread_label.pack(fill=tk.X, padx=10, pady=5)
        
        # Legend
        legend_frame = tk.LabelFrame(
            agent_frame,
            text="Legend",
            font=self.fonts['header'],
            bg=self.colors['panel'],
            fg=self.colors['text'],
            bd=2
        )
        legend_frame.pack(fill=tk.X, padx=20, pady=10)
        
        legend_items = [
            ("🔵 Passive Orders", self.colors['passive_order']),
            ("🟢 Bid Side", self.colors['bid']),
            ("🔴 Ask Side", self.colors['ask'])
        ]
        
        for text, color in legend_items:
            label = tk.Label(
                legend_frame,
                text=text,
                font=self.fonts['data'],
                bg=color,
                fg=self.colors['text'],
                anchor='w'
            )
            label.pack(fill=tk.X, padx=10, pady=2)
    
    def update_data(self, market_data: MarketData):
        """Thread-safe data update"""
        self.data_queue.put(market_data)
    
    def start_update_thread(self):
        """Start the GUI update thread"""
        def update_loop():
            while self.running:
                try:
                    # Process all queued updates
                    while not self.data_queue.empty():
                        market_data = self.data_queue.get_nowait()
                        self._update_gui(market_data)
                    
                    time.sleep(0.1)  # 10 FPS
                except Exception as e:
                    print(f"GUI update error: {e}")
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
    
    def _update_gui(self, data: MarketData):
        """Update GUI elements with new data"""
        try:
            # Update position panel
            position_color = '#00ff00' if data.position > 0 else '#ff0000' if data.position < 0 else self.colors['text']
            pnl_color = '#00ff00' if data.pnl > 0 else '#ff0000' if data.pnl < 0 else self.colors['text']
            
            self.position_label.config(text=f"POSITION: {data.position:.1f}", fg=position_color)
            self.pnl_label.config(text=f"P&L: ${data.pnl:.2f}", fg=pnl_color)
            self.cash_label.config(text=f"CASH: ${data.cash:.2f}")
            self.step_label.config(text=f"STEP: {data.step}")
            self.action_label.config(text=f"LAST ACTION: {data.last_action}")
            
            # Update DOM ladder
            self._update_dom_ladder(data)
            
            # Update agent panel
            self._update_agent_panel(data)
            
        except Exception as e:
            print(f"GUI update error: {e}")
    
    def _update_dom_ladder(self, data: MarketData):
        """Update the DOM ladder display"""
        # Calculate best bid/ask for spread
        best_bid = data.bids[0, 0] if len(data.bids) > 0 else 0
        best_ask = data.asks[0, 0] if len(data.asks) > 0 else 0
        spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0
        
        self.mid_price_label.config(text=f"MID: ${data.mid_price:.2f}")
        self.spread_label.config(text=f"SPREAD: ${spread:.3f}")
        
        # Update each row
        for i in range(self.levels):
            row_labels = self.dom_rows[i]
            
            # Bid data (left side)
            if i < len(data.bids):
                bid = data.bids[i]
                row_labels[0].config(text=f"{bid[2]:.1f}")  # volume
                row_labels[1].config(text=f"{bid[1]:.1f}")  # size
                row_labels[2].config(text=f"{bid[0]:.2f}")  # price
            else:
                row_labels[0].config(text="")
                row_labels[1].config(text="")
                row_labels[2].config(text="")
            
            # Ask data (right side)
            if i < len(data.asks):
                ask = data.asks[i]
                row_labels[3].config(text=f"{ask[0]:.2f}")  # price
                row_labels[4].config(text=f"{ask[1]:.1f}")  # size
                row_labels[5].config(text=f"{ask[2]:.1f}")  # volume
            else:
                row_labels[3].config(text="")
                row_labels[4].config(text="")
                row_labels[5].config(text="")
            
            # Check for passive orders at this level
            order_text = ""
            order_bg = self.colors['panel']
            
            # Check bid orders
            if (data.passive_bid_order and i < len(data.bids) and 
                abs(data.bids[i, 0] - data.passive_bid_order[0]) < 0.001):
                order_text += f"BID {data.passive_bid_order[1]:.1f} "
                order_bg = self.colors['passive_order']
            
            # Check ask orders  
            if (data.passive_ask_order and i < len(data.asks) and 
                abs(data.asks[i, 0] - data.passive_ask_order[0]) < 0.001):
                order_text += f"ASK {data.passive_ask_order[1]:.1f}"
                order_bg = self.colors['passive_order']
            
            row_labels[6].config(text=order_text, bg=order_bg)
    
    def _update_agent_panel(self, data: MarketData):
        """Update agent information panel"""
        # Working orders
        if data.passive_bid_order:
            bid_text = f"BID: {data.passive_bid_order[1]:.1f} @ ${data.passive_bid_order[0]:.2f}"
            self.bid_order_label.config(text=bid_text, fg=self.colors['passive_order'])
        else:
            self.bid_order_label.config(text="BID: None", fg=self.colors['text'])
        
        if data.passive_ask_order:
            ask_text = f"ASK: {data.passive_ask_order[1]:.1f} @ ${data.passive_ask_order[0]:.2f}"
            self.ask_order_label.config(text=ask_text, fg=self.colors['passive_order'])
        else:
            self.ask_order_label.config(text="ASK: None", fg=self.colors['text'])
    
    def run(self):
        """Start the GUI"""
        try:
            self.root.mainloop()
        finally:
            self.running = False
    
    def close(self):
        """Close the GUI"""
        self.running = False
        self.root.quit()
        self.root.destroy()