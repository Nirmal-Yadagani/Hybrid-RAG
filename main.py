def main():
    print("Hello from hybrid-rag!")
    
    from langchain_core.globals import set_debug

    from src.retriever import QAbot


    bot = QAbot()


    # set_debug(True)
    bot.answer(input('Query: ')) # type: ignore


if __name__ == "__main__":
    main()
