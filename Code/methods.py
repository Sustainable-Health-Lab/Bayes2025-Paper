import get_summaries 
import test_comparison 
import all_observations 
import get_example_hospital_results
from util import generate_n_file

generate_n_file('HospiceTreat')
get_summaries.main()
all_observations.main()
test_comparison.main()
get_example_hospital_results.main()

