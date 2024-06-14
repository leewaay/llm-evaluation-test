import pytest
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

# 고객이 배송 상태를 문의하는 경우
first_test_case = LLMTestCase(
    input="제품 배송 언제쯤 도착하나요?",
    actual_output="제품은 내일 배송될 예정입니다.",
    context=["고객님이 주문하신 제품의 배송 예정일은 내일입니다."]
)
# 고객이 반품 절차를 문의하는 경우, 챗봇이 관련 정보가 없음에도 불구하고 잘못된 정보를 제공하는 예
second_test_case = LLMTestCase(
    input="반품하려면 어떻게 해야 하나요?",
    actual_output="반품은 접수가 종료되었으므로 현재는 반품이 불가능합니다.",
    context=["고객은 웹사이트를 통해 언제든지 반품 접수를 할 수 있습니다."]
)

dataset = EvaluationDataset(test_cases=[first_test_case, second_test_case])

@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_customer_chatbot(test_case: LLMTestCase):
    hallucination_metric = HallucinationMetric(threshold=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [hallucination_metric, answer_relevancy_metric])