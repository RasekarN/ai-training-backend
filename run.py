from src.data_loader.yahoo_loader import YahooDataLoader
from src.engine.live_engine import LiveEngine
from src.engine.signal_formatter import SignalFormatter
from src.models.train_model import ModelTrainer

# 1. Load data
loader = YahooDataLoader()
df = loader.fetch("^NSEI", interval="1d", period="3y")

# 2. Train model
trainer = ModelTrainer()
model = trainer.train(df)

# 3. Start live engine
engine = LiveEngine(model)
formatter = SignalFormatter()

# 4. Generate signal
result = engine.analyze(df)

# 5. Show output
formatter.format_console(result)
