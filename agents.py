# Agent Orchestration Script with Azure OpenAI
import asyncio
from semantic_kernel.agents import Agent, ChatCompletionAgent, SequentialOrchestration
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatMessageContent


def get_agents() -> list[Agent]:
    """
    Create a list of agents: summarizer, classifier, and recommended action agents.
    """

    # -----------------------------
    # Initialize Azure Chat service
    # -----------------------------
    # REPLACE the placeholders below with your actual Azure OpenAI details
    azure_service = AzureChatCompletion(
        deployment_name="gpt-4o",         # e.g., "gpt-4-deployment"
        endpoint="https://project54194114-resource.cognitiveservices.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview",  # Full endpoint URL
        api_key="DxQLJ7Favlf6y1Pdd9pyJRUM1yCh4tgPYXy9eRXJodz5xxXiAYgxJQQJ99BHACHYHv6XJ3w3AAAAACOGphkM"                          # Your Azure OpenAI API key
    )

    # -----------------------------
    # Create a summarizer agent
    # -----------------------------
    summarizer_agent = ChatCompletionAgent(
        name="SummarizerAgent",
        instructions="""
        Summarize the customer's feedback in one short sentence. Keep it neutral and concise.
        Example output:
        App crashes during photo upload.
        User praises dark mode feature.
        """,
        service=azure_service,
    )

    # -----------------------------
    # Create a classifier agent
    # -----------------------------
    classifier_agent = ChatCompletionAgent(
        name="ClassifierAgent",
        instructions="""
        Classify the feedback as one of the following: Positive, Negative, or Feature request.
        """,
        service=azure_service,
    )

    # -----------------------------
    # Create a recommended action agent
    # -----------------------------
    action_agent = ChatCompletionAgent(
        name="ActionAgent",
        instructions="""
        Based on the summary and classification, suggest the next action in one short sentence.
        Example output:
        Escalate as a high-priority bug for the mobile team.
        Log as positive feedback to share with design and marketing.
        Log as enhancement request for product backlog.
        """,
        service=azure_service,
    )

    # Return all agents in a list
    return [summarizer_agent, classifier_agent, action_agent]


def agent_response_callback(message: ChatMessageContent) -> None:
    """
    Callback function to print responses from each agent during orchestration.
    """
    print(f"# {message.name}\n{message.content}")


async def main():
    # -----------------------------
    # Input task
    # -----------------------------
    task = """
    I tried updating my profile picture several times today, but the app kept freezing halfway through the process. 
    I had to restart it three times, and in the end, the picture still wouldn't upload. 
    It's really frustrating and makes the app feel unreliable.
    """

    # -----------------------------
    # Create sequential orchestration
    # -----------------------------
    sequential_orchestration = SequentialOrchestration(
        members=get_agents(),
        agent_response_callback=agent_response_callback,
    )

    # -----------------------------
    # Create and start runtime
    # -----------------------------
    runtime = InProcessRuntime()
    runtime.start()

    try:
        # -----------------------------
        # Invoke orchestration
        # -----------------------------
        orchestration_result = await sequential_orchestration.invoke(
            task=task,
            runtime=runtime,
        )

        # -----------------------------
        # Get final result
        # -----------------------------
        value = await orchestration_result.get(timeout=20)
        print(f"\n****** Task Input ******\n{task}")
        print(f"***** Final Result *****\n{value}")

    except Exception as e:
        print("Error invoking orchestration:", e)

    finally:
        # -----------------------------
        # Stop runtime
        # -----------------------------
        await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
