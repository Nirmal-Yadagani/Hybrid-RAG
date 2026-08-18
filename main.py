def main():
    print("Hello from hybrid-rag!")

    from src.retriever import QAbot


    bot = QAbot()

    bot.answer(input('Query: '))


if __name__ == "__main__":
    main()
