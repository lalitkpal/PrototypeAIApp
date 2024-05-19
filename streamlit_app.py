import streamlit as st
from StockSentimentsFromInternet import agent_executor


def stream_main():
    st.title("Stock Sentiment Analysis")
    ticker = st.text_input("Enter a stock ticker symbol:")
    time_period = st.selectbox("Enter time period: ", ("Today", "Weekly", "Monthly", "Yearly"))
    analyze_button = st.button("Analyze")

    if analyze_button:
        query = f"Analyze sentiment of {ticker} for time period {time_period}"
        result = agent_executor.invoke({"input": query})
        st.write(result)


if __name__ == '__main__':
    stream_main()
