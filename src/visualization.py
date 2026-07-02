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

def plot_close_price(df, ticker):
    """
    Plot Closing Price.
    """

    plt.figure(figsize=(12,6))

    plt.plot(
        df.index,
        df["Close"],
        color="blue",
        linewidth=2
    )

    plt.title(f"{ticker} Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Close Price")

    plt.grid(True)

    plt.show()

def plot_daily_return(df, ticker):
    """
    Plot Daily Return.
    """

    plt.figure(figsize=(12,6))

    plt.plot(
        df.index,
        df["Daily_Return"],
        color="green"
    )

    plt.title(f"{ticker} Daily Return")

    plt.xlabel("Date")

    plt.ylabel("Return")

    plt.grid(True)

    plt.show()