import openai
import json
import Levenshtein
from PIL import Image
import pandas as pd
import ast

client = openai.OpenAI()

from langchain.tools import tool
from pydantic import BaseModel
from typing import List, Dict
import json
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.output_parsers import PydanticOutputParser

### 1. Extract Order Details
class OrderItem(BaseModel):
    model: str
    quantity: int

class OrderDetails(BaseModel):
    customer: Dict[str, str]
    order_items: List[OrderItem]
        
order_parser = PydanticOutputParser(pydantic_object=OrderDetails)

@tool
def extract_order_details(text: str) -> OrderDetails:
    """Extract structured order data from a fax order."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """The following text is a fax order for hardware parts. Extract structured 
            order data from this order, specifically customer information and the products being ordered. Product names 
            will be written in Japanese. Product models will consist of English letters and numbers (eg SD-10
            or DXY). If product models are included in the order then do not include the product names in the output.
            However if the products are listed by their names only then do not attempt to include the product models 
            in the output. The company receiving the orders is called Best Parts, and so the customer name will always be 
            something other than Best Parts"""},
            {"role": "user", "content": text}
        ],
        functions=[
            {
                "name": "parse_order",
                "description": "Extracts structured order data from unstructured text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "address": {"type": "string"},
                                "contact_person": {"type": "string"},
                                "phone": {"type": "string"},
                                "fax": {"type": "string"}
                            },
                            "required": ["name", "address", "contact_person", "phone", "fax"]
                        },
                        "order_items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "product_name": {"type": "string"},
                                    "model": {"type": "string"},
                                    "quantity": {"type": "integer"}
                                },
                                "required": ["quantity"]
                            }
                        }
                    },
                    "required": ["customer", "order_items"]
                }
            }
        ],
        function_call={"name": "parse_order"}  # Explicitly call the function
    )

    function_args = response.choices[0].message.function_call.arguments
    print('FUNCTION ARGS')
    print(function_args)
    return function_args

class CustomerMatch(BaseModel):
    id: int
    name_one: str
    name_two: str

@tool
def find_best_customer(input_data: str) -> CustomerMatch:
    """Finds the best customer match for the detected name."""
    input_data = json.loads(input_data)
    detected_name = input_data["detected_name"]
    customer_list = input_data["customer_list"]
    customer_entries = [
        {
            "id": customer["node"]["id"],
            "name_one": customer["node"]["name_one"].strip(),
            "name_two": customer["node"]["name_two"].strip()
        }
        for customer in customer_list
    ]
    
    # Create a system message instructing GPT-4 Turbo to return structured JSON
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that identifies the best-matching customer from a list based on the detected name. "
                                          "You MUST return a JSON object exactly matching the format of the customer entries provided."},
            {"role": "user", "content": f"The detected customer name is: '{detected_name}'.\n"
                                        f"Here is a list of possible customer entries:\n"
                                        f"{json.dumps(customer_entries, ensure_ascii=False, indent=2)}\n"
                                        f"Return ONLY the best-matching entry as a valid JSON object with keys: id, name_one, name_two. DO NOT include any explanation or extra text."}
        ],
        response_format={"type": "json_object"}
    )

    # Parse the response JSON
    best_match = json.loads(response.choices[0].message.content)

    return best_match

### 3. Find Best Shipping Address
class AddressMatch(BaseModel):
    id: int
    name_one: str
    name_two: str
    phone: str
    fax: str
    address_city: str
    address_street: str

@tool
def find_best_shipping(input_data: str) -> AddressMatch:
    """Finds the best matching shipping address."""
    if isinstance(input_data, str):
        input_data = json.loads(input_data)
    detected_address = input_data["detected_address"]
    address_list = input_data["address_list"]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that identifies the best-matching address from a list based on the detected address. "
                                          "You MUST return a JSON object exactly matching the format of the address entries provided."},
            {"role": "user", "content": f"The detected address is: '{detected_address}'.\n"
                                        f"Here is a list of possible address entries:\n"
                                        f"{json.dumps(address_list, ensure_ascii=False, indent=2)}\n"
                                        f"Return ONLY the best-matching entry as a valid JSON object with keys: id, name_one, name_two, phone, fax, address_city, address_street.\n"
                                        f"DO NOT include any explanation or extra text."}
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

def get_fax_info(dtd):
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Define the agent
    agent = initialize_agent(
        tools=[extract_order_details, find_best_customer, find_best_shipping],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Agent decides which tool to use
        verbose=True
    )

    function_args = agent.run(f"""Extract order details from: {dtd}. Return only the hash from the tool
    and no additional text""")
    order_details = json.loads(function_args)

    detected_customer = order_details['customer']['name']
    customers_list = sorted(customers_result, 
                                key=lambda x: Levenshtein.distance(detected_customer, 
                                                                    f"{x['node']['name_one']} {x['node']['name_two']}"))
    customers_list = customers_list[:10]
    customers_list

    input_data = {'detected_name':detected_customer,
                'customer_list':customers_list}
    best_customer_match = json.loads(agent.run(f"""find the best customer with a detected customer and custsomer list 
    of {json.dumps(input_data)}. Return only the customer entry within the list that best
    matches the detected customer as a valid JSON object and do not include any additional text."""))
    best_customer_match

    customer_info = str(order_details['customer'])
    best_customer_id = best_customer_match['id']
    address_list = sorted(shippings_result, 
                            key=lambda x: Levenshtein.distance(customer_info, 
                                                                f"{x['node']['name_one']} {x['node']['name_two']} {x['node']['phone']} {x['node']['fax']} {x['node']['address_city']} {x['node']['address_street']}"))
    address_list_cid = [x for x in address_list if x['node']['customer_id']==best_customer_id]
    if len(address_list_cid) > 0:
        address_list = address_list_cid
    address_list = address_list[:10]
    input_data = {"detected_address":customer_info, 
                "address_list":address_list}
    input_data["detected_address"] = ast.literal_eval(input_data["detected_address"])
    best_address_match = agent.run(f"""find the best shipping info match with detected address and 
    shipping address list of {json.dumps(input_data, ensure_ascii=False)}. Return only the shipping address entry within the list 
    that best matches the detected address and do not include any additional text.""")

    #get products
    product_matches = []
    for x in order_details['order_items']:
        if 'model' in x:
            detected_product = x['model']
            products_list = sorted(products_result, 
                                    key=lambda x: Levenshtein.distance(detected_product, 
                                                                        f"{x['node']['code']}"))
            product_matches.append(products_list[0]['node'])
        elif 'product_name' in x:
            detected_product = x['product_name']
            products_list = sorted(products_result, 
                                    key=lambda x: Levenshtein.distance(detected_customer, 
                                                                        f"{x['node']['name']}"))
            product_matches.append(products_list[0]['node'])

    return best_customer_match, best_address_match, product_matches

if __name__ == "__main__":
    #normally you would get this by applying get_text_from_image on an image
    #but this requires google cloud credentials so just using a test file
    with open('document_text_data_test.json', 'r') as f:
        dtd = json.load(f)
    customers = pd.read_csv('customers.csv')
    customers_result = [{'node':{'name_one':row['Name1'], 
                                    'name_two':row['Name2'], 
                                    'id':row['ID']}} for index, row in customers.iterrows()]
    shippings_result = [{'node':{'name_one':row['Name1'],
        'name_two':row['Name2'],
        'id':row['ID'],
            'customer_id':row['ID'],
            'fax':row['Fax'],
            'phone':row['Phone'],
            'address_city':row['Address_City'],
            'address_street':row['Address_Street']}} for index, row in customers.iterrows()]
    products = pd.read_csv('products.csv')
    products['id'] = range(100)
    products_result = [{'node':{'code':row['Part Number'],
                                'name':row['Part Name'],
                                'id':row['id']}} for index, row in products.iterrows()]
    best_customer_match, best_address_match, product_matches = get_fax_info(dtd)
    print('BEST CUSTOMER MATCH')
    print(best_customer_match)
    print('BEST ADDRESS MATCH')
    print(best_address_match)
    print('PRODUCT MATCHES')
    print(product_matches)
