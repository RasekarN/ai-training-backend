import json

class SignalFormatter:

    def format_console(self, result):
        print("------ LIVE SIGNAL ------")
        print(f"Signal: {result['signal']}")
        print(f"ML Probability Up: {result['ml_prob_up']:.2f}")
        print(f"ML Probability Down: {result['ml_prob_down']:.2f}")
        print(f"Latest Close: {result['latest_close']}")
        print("--------------------------")

    def format_json(self, result):
        return json.dumps(result, indent=4)

    def format_webhook(self, result):
        return {
            "signal": result["signal"],
            "probability": result["ml_prob_up"],
            "price": result["latest_close"]
        }
