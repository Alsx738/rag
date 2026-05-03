import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver

from agents.graph import create_agent
from utility.auth import setup_users_table, login, signup

load_dotenv()


def _get_pg_connection_string() -> str:
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def _auth_flow() -> tuple[int, str]:
    print("\n1) Login")
    print("2) Signup")
    while True:
        choice = input("Choice > ").strip()
        if choice in ("1", "2"):
            break
        print("Enter 1 or 2.")

    username = input("Username > ").strip()
    password = input("Password > ").strip()

    if choice == "2":
        try:
            user_id = signup(username, password)
            print(f"Account created! Welcome, {username}.")
            return user_id, username
        except Exception:
            print("Username already taken. Try logging in.")
            return _auth_flow()
    else:
        user_id = login(username, password)
        if user_id is None:
            print("Wrong username or password.")
            return _auth_flow()
        print(f"Welcome back, {username}!")
        return user_id, username


def main():
    print("=" * 50)
    print(" Welcome to your personalized Amazon RAG!")
    print("=" * 50)

    setup_users_table()

    user_id, username = _auth_flow()
    thread_id = str(user_id)

    conn_string = _get_pg_connection_string()
    with PostgresSaver.from_conn_string(conn_string) as checkpointer:
        checkpointer.setup()
        agent = create_agent(checkpointer=checkpointer)

        print("\nType your search (e.g. 'looking for healthy food for my cat')")
        print("Type 'exit' to quit.\n")

        # Pass both thread id and username to the agent config so checkpoints and
        # conversational context can include identity info when available.
        # Note: username here is not pulled from DB again; reuse the last login input.
        # The orchestrator/synthesizer can use this to answer personal queries.
        # include the authenticated username so checkpoint context can store identity
        config = {"configurable": {"thread_id": thread_id, "username": username}}

        while True:
            try:
                query = input("User > ")
                if query.lower() in ("exit", "quit", "q"):
                    print("Goodbye!")
                    break

                if not query.strip():
                    continue

                print("[Finder and Recommender agents are searching...]\n")

                # include the latest user message as a HumanMessage and, when available,
                # add the username into the configurable portion so it is part of the
                # saved thread context
                config["configurable"]["username"] = username
                result = agent.invoke({"messages": [HumanMessage(content=query)]}, config=config)
                final_message = result["messages"][-1].content
                print(f"SYNTHESIZER > {final_message}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break


if __name__ == "__main__":
    main()
