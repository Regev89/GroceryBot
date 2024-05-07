from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
import GroceryModelFit
import SpacyPosTagger
import ShoppingListLogger
import SingularFormTransformer

#Warp NLShoppingListService as Flask
from flask import Flask, request


def create_app(remote=False):

    app = Flask(__name__)

    # Initilize variables for langchain
    class Grocery(BaseModel):
        def __init__(self, grocery, amount, unit, action):
            self.grocery = grocery
            self.amount = amount
            self.unit = unit
            self.action = action
    class Grocerylist(BaseModel):
        groceries: list[Grocery] = Field(description="Grocery items as a list")


    GROQ_API_KEY_RP2 = <Enter your groq API>

    CHAT_GPT_4_API_KEY = <Enter OpenAI API>

    logger = ShoppingListLogger.setup_logger()

    chat = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY_RP2,
                    model_name="mixtral-8x7b-32768")  # .with_structured_output(Grocerylist, method="json_mode")

    system = """
 
you will receive a sentence a person has said.
first, identify the keywords, entities, and actions in the sentence.
    
if the keywords, entities, and actions are strongly tied to a purchasing groceries list, make note of that, and then,
    make a list of the groceries, their amounts, their amount unit, and the action related to each grocery.
    The actions related to grocery can be only one of the following:
    'add' - if the user asks to add the grocery to the purchasing groceries list.
    'delete'  - if the user asks to delete the grocery from the purchasing groceries list
    'update'  - if the user asks to update the grocery amount in the purchasing groceries list.
    'subtract' - if the user asks to subtract an amount from the grocery amount in the purchasing groceries list.
      
Notice the difference between remove all (as in delete) vs remove 2 (as in subtract 2).

If the input sentence indicates that a grocery amount needs to change, its related action should be 'update'. 

If the input sentence indicates that a grocery is missing or ran out, its related action should be 'add'.
    
If the input sentence indicates one has enough of the grocery, its related action should be 'delete'.
    
Units should be one of the following: g (for gram), Kg (for kilogram or kilo), Litre, or unit according to the product. no other measurements are allowed.
    
The list should contain the grocery in its singular form. If the grocery is in its plural form, turn it into its singular form. For example, if the grocery value is 'apples' turn it to 'apple'.
    
If the amount value is None or 0 then set it to 1.
    
If the amount value is a negative number, set it to its absolute number.    

If the unit is None, set it to 'unknown'.
    
Answer only with a valid JSON output:
    
    class Grocery(BaseModel) :
        def __init__(self, grocery, amount, unit,action) :
            self.grocery = grocery
            self.amount = amount
            self.unit = unit
            self.action = action
    class Grocerylist(BaseModel):
        groceries: list[Grocery]  = Field(description="Grocery items as a list")
 
    
    Don't explain your answer.   
    """

    human = "{text}"
    parser = JsonOutputParser(pydantic_object=Grocerylist)
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    initialize_app(remote)
    def post_process_llm_resp(groceries_lst):
        ret_val = []
        for item in groceries_lst:
            if item['grocery'] not in ('grocery','unknown'):
                sent = item['action'] + ' the ' + item['grocery']
                if SpacyPosTagger.has_noun(sent):
                    item['grocery'] = SingularFormTransformer.to_singular(item['grocery'])
                    ret_val.append(item)
        return ret_val

    def add_response_meta_data(type,resp):
        match type:
            case 0:
                resp['valid'] = True
                resp['error_code'] = 0
            case 100:
                resp['valid'] = False
                resp['reason'] = 'LLM response is not a valid json'
                resp['error_code'] = 100
            case 200:
                resp['valid'] = False
                resp['reason'] = 'Incomplete grocery sentence'
                resp['error_code'] = 200
            case 300:
                resp['valid'] = False
                resp['reason'] = 'Invalid shopping list request'
                resp['error_code'] = 300

    @app.before_request
    def log_request():
        if request.method == 'GET':
            user_ip = request.remote_addr
            logger.info("Received GET request for %s from %s with args: %s", request.path,request.remote_addr, request.args.to_dict())

    @app.route('/analyze_nl_request', methods=['GET'])
    def analyze_natuarl_spl_req():
        text = request.args.get('nl_request', 'default')

        ret_val = {}
        grocery = GroceryModelFit.is_grocery_sentence(text)
        if (grocery):
            #check existence of nouns to prevent incomplete sentences like "please add 3"
            #We do this check since we found out that for such sentences the LLM return a make up response
            # such as 'add 5 apples' or 'add 2 bananas'
            # Although it is not the best thing to do we check both noun and adj
            # since there are groceries that may have an adj pos (like orange).
            if SpacyPosTagger.has_noun_or_adj(text):
                logger.info(f"The nl_request is a grocery sentence")
                chain = prompt | chat | parser
                try:
                    response = chain.invoke({"text": text})
                    logger.info("LLM response:" + str(response))
                    # Since LLM response is not stable and may return list instead of dict
                    # then check response instance before we preceded
                    response = response[0] if isinstance(response, list) else response
                    # use post_process_llm_resp to clean groceries from invalid actions
                    # Since LLM is not stable , may return 'groceries' or grocerylist.groceries ,
                    # overcome that with single line if-else
                    ret_val['groceries'] = post_process_llm_resp(response['grocerylist']['groceries'] if 'grocerylist' in response else  response['groceries'] )

                    if(len(ret_val['groceries'])>0):
                    #Add metafields
                        add_response_meta_data(0,ret_val)
                    else:
                        logger.info(f"The nl_request is an INCOMPLETE grocery sentence")
                        add_response_meta_data(200, ret_val)


                    logger.info("Value returned to the user: %s", ret_val)
                    return ret_val
                except Exception as e:
                    add_response_meta_data(100, ret_val)
                    logger.info("Invalid json exception: %s",str(e))
                    logger.info("Value returned to the user: %s", ret_val)
                    return ret_val
            else:
                logger.info(f"The nl_request is an INCOMPLETE grocery sentence")
                add_response_meta_data(200, ret_val)
        else:
         logger.info(f"The nl_request is NOT a grocery sentence")
         add_response_meta_data(300, ret_val)

        logger.info("Value returned to the user: %s", ret_val)
        return ret_val

    return app
def initialize_app(remote):
    print("Initializing the application...")
    GroceryModelFit.init_grocery_model()

    print("Initialization complete.")

if __name__ == '__main__':
    app = create_app()

    #for local service
    app.run(debug=False)

    #for www service
    #app.run(host='0.0.0.0', port=5001,debug=False)
