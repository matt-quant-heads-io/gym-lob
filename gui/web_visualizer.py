import yaml
import torch
import json
import time
import threading
import webbrowser
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from envs import OrderbookEnv
from model import ActorCritic

class WebVisualizer:
    """Professional web-based DOM visualization"""
    
    def __init__(self, model_path: str, config_path: str, port: int = 8080):
        self.port = port
        self.model_path = model_path
        self.config_path = config_path
        
        # Load configuration and model
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.env = OrderbookEnv(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = ActorCritic(
            input_size=self.env.observation_space.shape[0],
            lstm_hidden_size=self.config['model']['lstm_hidden_size'],
            transformer_heads=self.config['model']['transformer_heads'],
            transformer_layers=self.config['model']['transformer_layers'],
            action_dim=self.env.action_space.n,
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.action_names = [
            "DO_NOTHING", "PASSIVE_BUY", "PASSIVE_SELL", 
            "CANCEL_ORDERS", "MARKET_BUY", "MARKET_SELL", "CLOSE_ALL"
        ]
        
        self.current_data = {
            'bids': [], 'asks': [], 'mid_price': 0, 'position': 0,
            'cash': 0, 'pnl': 0, 'step': 0, 'last_action': 'INITIALIZING',
            'passive_bid_order': None, 'passive_ask_order': None,
            'episode': 0, 'total_episodes': 0, 'status': 'Loading...'
        }
        self.running = False
        self.httpd = None
    
    def create_web_files(self):
        """Create enhanced HTML interface"""
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LOB Trading Agent - Professional DOM Visualizer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
            background: linear-gradient(135deg, #0c0c0c 0%, #1a1a1a 100%);
            color: #e0e0e0;
            min-height: 100vh;
            overflow-x: auto;
        }
        
        .container { 
            max-width: 1600px; 
            margin: 0 auto; 
            padding: 20px;
        }
        
        .header { 
            text-align: center; 
            margin-bottom: 30px;
            background: linear-gradient(90deg, #2d2d2d, #3a3a3a);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        .header h1 { 
            font-size: 2.2em; 
            margin-bottom: 10px;
            background: linear-gradient(45deg, #00ff88, #00ccff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .metrics { 
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric { 
            background: linear-gradient(135deg, #2d2d2d, #3a3a3a);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            border: 1px solid #404040;
            transition: transform 0.2s ease;
        }
        
        .metric:hover { transform: translateY(-2px); }
        
        .metric .label { 
            font-size: 0.9em; 
            color: #999;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .metric .value { 
            font-size: 1.8em; 
            font-weight: bold;
            transition: color 0.3s ease;
        }
        
        .main-content { 
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 30px;
        }
        
        .dom-section { 
            background: linear-gradient(135deg, #1e1e1e, #2a2a2a);
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            border: 1px solid #404040;
        }
        
        .dom-section h3 { 
            margin-bottom: 20px;
            font-size: 1.4em;
            color: #00ff88;
            text-align: center;
            padding-bottom: 10px;
            border-bottom: 2px solid #333;
        }
        
        .dom-header { 
            display: grid; 
            grid-template-columns: 1fr 1fr 1.2fr 1.2fr 1fr 1fr 1.5fr;
            gap: 8px;
            margin-bottom: 15px;
            padding: 12px 8px;
            background: #333;
            border-radius: 5px;
            font-weight: bold;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .dom-rows { 
            max-height: 600px;
            overflow-y: auto;
            scrollbar-width: thin;
            scrollbar-color: #555 #2a2a2a;
        }
        
        .dom-row { 
            display: grid; 
            grid-template-columns: 1fr 1fr 1.2fr 1.2fr 1fr 1fr 1.5fr;
            gap: 8px;
            margin: 3px 0;
            padding: 8px;
            border-radius: 4px;
        }
        
        .dom-cell { 
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        
        .bid-cell { 
            background: linear-gradient(135deg, #0d4f3c, #1a6b4f);
            text-align: right;
            border-left: 3px solid #00ff88;
        }
        
        .ask-cell { 
            background: linear-gradient(135deg, #4a1e1e, #6b2a2a);
            text-align: left;
            border-right: 3px solid #ff4444;
        }
        
        .order-cell { 
            background: #333;
            text-align: center;
            font-weight: bold;
        }
        
        .passive-order { 
            background: linear-gradient(135deg, #ff6b35, #ff8c42) !important;
            color: #000 !important;
            font-weight: bold;
            animation: pulse 2s infinite;
            box-shadow: 0 0 10px rgba(255, 107, 53, 0.5);
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        .side-panel { 
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .panel-section { 
            background: linear-gradient(135deg, #1e1e1e, #2a2a2a);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            border: 1px solid #404040;
        }
        
        .panel-section h4 { 
            margin-bottom: 15px;
            color: #00ccff;
            font-size: 1.1em;
            padding-bottom: 8px;
            border-bottom: 1px solid #333;
        }
        
        .panel-item { 
            margin: 10px 0;
            padding: 8px 12px;
            background: #333;
            border-radius: 5px;
            font-size: 0.9em;
        }
        
        .working-order { 
            background: linear-gradient(90deg, #2a2a2a, #3a3a3a);
            border-left: 4px solid #ff6b35;
            color: #ff6b35;
            font-weight: bold;
        }
        
        .positive { color: #00ff88; }
        .negative { color: #ff4444; }
        .neutral { color: #e0e0e0; }
        
        .connection-status { 
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }
        
        .connected { background: #00ff88; color: #000; }
        .disconnected { background: #ff4444; color: #fff; }
        .connecting { background: #ffaa00; color: #000; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 LOB Trading Agent</h1>
            <h2>Professional DOM Visualizer</h2>
            <div id="connection-status" class="connection-status connecting">🔄 Connecting...</div>
        </div>
        
        <div class="metrics">
            <div class="metric">
                <div class="label">📊 Position</div>
                <div class="value neutral" id="position">0.0</div>
            </div>
            <div class="metric">
                <div class="label">💰 P&L</div>
                <div class="value neutral" id="pnl">$0.00</div>
            </div>
            <div class="metric">
                <div class="label">💵 Cash</div>
                <div class="value" id="cash">$0.00</div>
            </div>
            <div class="metric">
                <div class="label">⏱️ Step</div>
                <div class="value" id="step">0</div>
            </div>
            <div class="metric">
                <div class="label">📈 Mid Price</div>
                <div class="value" id="mid-price">$0.00</div>
            </div>
            <div class="metric">
                <div class="label">📏 Spread</div>
                <div class="value" id="spread">$0.00</div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="dom-section">
                <h3>📊 Order Book Ladder</h3>
                <div class="dom-header">
                    <div>BID VOL</div>
                    <div>BID SIZE</div>
                    <div>BID PRICE</div>
                    <div>ASK PRICE</div>
                    <div>ASK SIZE</div>
                    <div>ASK VOL</div>
                    <div>ORDERS</div>
                </div>
                <div class="dom-rows" id="dom-rows"></div>
            </div>
            
            <div class="side-panel">
                <div class="panel-section">
                    <h4>🤖 Agent Status</h4>
                    <div class="panel-item">
                        <strong>Last Action:</strong> <span id="last-action">INITIALIZING</span>
                    </div>
                </div>
                
                <div class="panel-section">
                    <h4>🔵 Working Orders</h4>
                    <div class="panel-item" id="bid-order">BID: None</div>
                    <div class="panel-item" id="ask-order">ASK: None</div>
                </div>
                
                <div class="panel-section">
                    <h4>📈 Market Statistics</h4>
                    <div class="panel-item">Best Bid: <span id="best-bid">$0.00</span></div>
                    <div class="panel-item">Best Ask: <span id="best-ask">$0.00</span></div>
                    <div class="panel-item">Total Bid Vol: <span id="total-bid-vol">0</span></div>
                    <div class="panel-item">Total Ask Vol: <span id="total-ask-vol">0</span></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let isConnected = false;
        
        function updateDisplay(data) {
            if (!data) return;
            
            // Update connection status
            if (!isConnected) {
                isConnected = true;
                document.getElementById('connection-status').textContent = '🟢 Connected';
                document.getElementById('connection-status').className = 'connection-status connected';
            }
            
            // Update metrics
            const positionEl = document.getElementById('position');
            positionEl.textContent = data.position.toFixed(1);
            positionEl.className = 'value ' + (data.position > 0 ? 'positive' : data.position < 0 ? 'negative' : 'neutral');
            
            const pnlEl = document.getElementById('pnl');
            pnlEl.textContent = '$' + data.pnl.toFixed(2);
            pnlEl.className = 'value ' + (data.pnl > 0 ? 'positive' : data.pnl < 0 ? 'negative' : 'neutral');
            
            document.getElementById('cash').textContent = '$' + data.cash.toFixed(2);
            document.getElementById('step').textContent = data.step;
            document.getElementById('mid-price').textContent = '$' + data.mid_price.toFixed(2);
            document.getElementById('last-action').textContent = data.last_action;
            
            // Calculate and display spread
            const spread = data.asks.length > 0 && data.bids.length > 0 ? 
                (data.asks[0][0] - data.bids[0][0]) : 0;
            document.getElementById('spread').textContent = '$' + spread.toFixed(3);
            
            // Update market statistics
            if (data.bids.length > 0) {
                document.getElementById('best-bid').textContent = '$' + data.bids[0][0].toFixed(2);
                const totalBidVol = data.bids.reduce((sum, bid) => sum + bid[2], 0);
                document.getElementById('total-bid-vol').textContent = totalBidVol.toFixed(0);
            }
            
            if (data.asks.length > 0) {
                document.getElementById('best-ask').textContent = '$' + data.asks[0][0].toFixed(2);
                const totalAskVol = data.asks.reduce((sum, ask) => sum + ask[2], 0);
                document.getElementById('total-ask-vol').textContent = totalAskVol.toFixed(0);
            }
            
            // Update DOM
            updateOrderBook(data);
            
            // Update working orders
            updateWorkingOrders(data);
        }
        
        function updateOrderBook(data) {
            const domRows = document.getElementById('dom-rows');
            domRows.innerHTML = '';
            
            const maxLevels = Math.max(data.bids.length, data.asks.length, 15);
            for (let i = 0; i < maxLevels; i++) {
                const row = document.createElement('div');
                row.className = 'dom-row';
                
                const bid = i < data.bids.length ? data.bids[i] : [0, 0, 0];
                const ask = i < data.asks.length ? data.asks[i] : [0, 0, 0];
                
                // Check for passive orders
                let orderText = '';
                let hasPassiveOrder = false;
                
                if (data.passive_bid_order && bid[0] && Math.abs(bid[0] - data.passive_bid_order[0]) < 0.001) {
                    orderText = '🔵 BID ' + data.passive_bid_order[1].toFixed(1);
                    hasPassiveOrder = true;
                }
                
                if (data.passive_ask_order && ask[0] && Math.abs(ask[0] - data.passive_ask_order[0]) < 0.001) {
                    orderText += (orderText ? ' ' : '') + '🔵 ASK ' + data.passive_ask_order[1].toFixed(1);
                    hasPassiveOrder = true;
                }
                
                const passiveClass = hasPassiveOrder ? ' passive-order' : '';
                
                row.innerHTML = `
                    <div class="dom-cell bid-cell${passiveClass}">${bid[2] ? bid[2].toFixed(1) : ''}</div>
                    <div class="dom-cell bid-cell${passiveClass}">${bid[1] ? bid[1].toFixed(1) : ''}</div>
                    <div class="dom-cell bid-cell${passiveClass}">${bid[0] ? bid[0].toFixed(2) : ''}</div>
                    <div class="dom-cell ask-cell${passiveClass}">${ask[0] ? ask[0].toFixed(2) : ''}</div>
                    <div class="dom-cell ask-cell${passiveClass}">${ask[1] ? ask[1].toFixed(1) : ''}</div>
                    <div class="dom-cell ask-cell${passiveClass}">${ask[2] ? ask[2].toFixed(1) : ''}</div>
                    <div class="dom-cell order-cell${passiveClass}">${orderText}</div>
                `;
                
                domRows.appendChild(row);
            }
        }
        
        function updateWorkingOrders(data) {
            const bidOrderEl = document.getElementById('bid-order');
            const askOrderEl = document.getElementById('ask-order');
            
            if (data.passive_bid_order) {
                bidOrderEl.textContent = `BID: ${data.passive_bid_order[1].toFixed(1)} @ $${data.passive_bid_order[0].toFixed(2)}`;
                bidOrderEl.className = 'panel-item working-order';
            } else {
                bidOrderEl.textContent = 'BID: None';
                bidOrderEl.className = 'panel-item';
            }
            
            if (data.passive_ask_order) {
                askOrderEl.textContent = `ASK: ${data.passive_ask_order[1].toFixed(1)} @ $${data.passive_ask_order[0].toFixed(2)}`;
                askOrderEl.className = 'panel-item working-order';
            } else {
                askOrderEl.textContent = 'ASK: None';
                askOrderEl.className = 'panel-item';
            }
        }
        
        function handleConnectionError() {
            isConnected = false;
            document.getElementById('connection-status').textContent = '🔴 Disconnected';
            document.getElementById('connection-status').className = 'connection-status disconnected';
        }
        
        function pollData() {
            fetch('/data.json')
                .then(response => {
                    if (!response.ok) throw new Error('Network response was not ok');
                    return response.json();
                })
                .then(data => updateDisplay(data))
                .catch(error => handleConnectionError());
        }
        
        // Start polling at 10 FPS
        setInterval(pollData, 100);
        pollData(); // Initial load
    </script>
</body>
</html>'''
        
        # Create web directory if it doesn't exist
        os.makedirs('web', exist_ok=True)
        
        # Write the HTML file
        with open('web/index.html', 'w') as f:
            f.write(html_content)
        
        print(f"📁 Created web interface at: web/index.html")
    
    def start_web_server(self):
        """Start HTTP server for web interface"""
        class CustomHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, visualizer=None, **kwargs):
                self.visualizer = visualizer
                super().__init__(*args, **kwargs)
            
            def do_GET(self):
                if self.path == '/' or self.path == '/index.html':
                    self.path = '/web/index.html'
                elif self.path == '/data.json':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    self.wfile.write(json.dumps(self.visualizer.current_data).encode())
                    return
                
                super().do_GET()
            
            def log_message(self, format, *args):
                # Suppress server logs for cleaner output
                pass
        
        handler = lambda *args, **kwargs: CustomHandler(*args, visualizer=self, **kwargs)
        
        try:
            self.httpd = HTTPServer(('localhost', self.port), handler)
            server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            server_thread.start()
            return True
        except OSError as e:
            if e.errno == 48:  # Address already in use
                print(f"⚠️  Port {self.port} is already in use. Trying port {self.port + 1}...")
                self.port += 1
                return self.start_web_server()
            else:
                print(f"❌ Failed to start web server: {e}")
                return False
    
    def run(self, episodes=5, step_delay=0.2, open_browser=True):
        """Run web visualization"""
        print("🌐 Starting Professional Web DOM Visualizer")
        print(f"🎯 Episodes: {episodes}")
        print(f"⏱️  Step Delay: {step_delay}s")
        print(f"🖥️  Device: {self.device}")
        
        # Create web files
        self.create_web_files()
        
        # Start HTTP server
        if not self.start_web_server():
            print("❌ Failed to start web server")
            return
        
        url = f'http://localhost:{self.port}'
        print(f"🌐 Web server started at: {url}")
        
        # Open browser
        if open_browser:
            print("🌐 Opening browser...")
            webbrowser.open(url)
        else:
            print(f"📝 Open your browser and navigate to: {url}")
        
        print("\n" + "="*60)
        print("🚀 VISUALIZATION STARTED")
        print("="*60)
        print(f"📊 Watch the DOM ladder in your browser at: {url}")
        print(f"🔵 Passive orders will be highlighted in orange")
        print(f"📈 Real-time P&L and position tracking")
        print(f"⏱️  Updates at 10 FPS for smooth visualization")
        print("="*60)
        
        # Update current data with initial values
        self.current_data.update({
            'total_episodes': episodes,
            'status': 'Running'
        })
        
        # Run episodes
        for episode in range(1, episodes + 1):
            print(f"\n🎮 Episode {episode}/{episodes}")
            self.current_data['episode'] = episode
            self.run_episode(episode_steps=1000, step_delay=step_delay)
            
            if episode < episodes:
                print("⏸️  Pausing between episodes...")
                time.sleep(2)
        
        print(f"\n🏁 All {episodes} episodes completed!")
        print(f"🌐 Web interface remains active at: {url}")
        print("Press Ctrl+C to stop the server")
        
        # Keep server running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down web server...")
            if self.httpd:
                self.httpd.shutdown()
    
    def run_episode(self, episode_steps=1000, step_delay=0.2):
        """Run episode and update web data"""
        obs, _ = self.env.reset()
        hidden = self.model.init_hidden(1, self.device)
        
        for step in range(episode_steps):
            # Get action
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action, _, _, _ = self.model.get_action_and_value(obs_tensor, hidden)
                action = action.item()
            
            # Execute
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Update web data
            self.current_data.update({
                'bids': self.env.current_bids.tolist(),
                'asks': self.env.current_asks.tolist(),
                'mid_price': float(self.env.mid_price),
                'position': float(self.env.position),
                'cash': float(self.env.cash),
                'pnl': float(info['pnl']),
                'step': step,
                'last_action': self.action_names[action],
                'passive_bid_order': [float(self.env.passive_order_manager.bid_order.price), 
                                    float(self.env.passive_order_manager.bid_order.size)] 
                                    if self.env.passive_order_manager.bid_order else None,
                'passive_ask_order': [float(self.env.passive_order_manager.ask_order.price), 
                                    float(self.env.passive_order_manager.ask_order.size)] 
                                    if self.env.passive_order_manager.ask_order else None,
            })
            
            if done:
                print(f"✅ Episode completed at step {step} | P&L: ${info['pnl']:.2f}")
                break
            
            time.sleep(step_delay)