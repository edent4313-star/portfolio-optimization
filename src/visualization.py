import matplotlib.pyplot as plt

def plot_closing_prices(assets):

    plt.figure(figsize=(14,6))

    for ticker, df in assets.items():

        plt.plot(
            df.index,
            df["Close"],
            label=ticker
        )

    plt.legend()

    plt.show()