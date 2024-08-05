import unittest
from langchain.schema import AgentAction, AgentFinish

from src.applications.text_to_sql import FlexibleOutputParser

class TestFlexibleOutputParser(unittest.TestCase):

    def setUp(self):
        self.parser = FlexibleOutputParser()

    def test_final_answer(self):
        text = "Here is the result. Final Answer: 42"
        result = self.parser.parse(text)
        self.assertIsInstance(result, AgentFinish)
        self.assertEqual(result.return_values['output'], "42")

    def test_action_parsing(self):
        text = """
        Thought: The data needs to be filtered before aggregation.
        Action: FilterData
        Action Input: {"filter": "age > 30"}
        """
        result = self.parser.parse(text)
        self.assertIsInstance(result, AgentAction)
        self.assertEqual(result.tool, "FilterData")
        self.assertEqual(result.tool_input, '{"filter": "age > 30"}')

    def test_unclear_output(self):
        text = "This is an unclear output without a clear pattern."
        result = self.parser.parse(text)
        self.assertIsInstance(result, AgentAction)
        self.assertEqual(result.tool, "human")
        self.assertEqual(result.tool_input, "Unclear output. Please clarify your next step.")

if __name__ == '__main__':
    unittest.main()