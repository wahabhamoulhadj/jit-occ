

# This variable is used to put the endpoint restful api url
issue_tracking_system_url = "issues.apache.org/jira"

# This variable is used to set software name
project_name = "samza"

# This variable is used to set software repository location in your hard drive
repository_path_location = "E:/CeleverDataset/samza" #"C:/Users/umroot/Downloads/kafka"

# The location for issue reports
bugs_report_path = "./issues/"

# The location for the SZZ algorithm
szz_results = "./results/"

# This variable is used to set the bug pattern inside issue reports. (e.g. "[a-z]+[-\t]+[0-9]+" is for Jira apache
# systems,"\#[0-9]+" for Github).
bug_pattern = "[a-z]+[-\t]+[0-9]+"

# This variable is used to set software repository branch
repository_branch = "master"

# Generalk settings
enable_parallel = True
max_results = 1_000
clone_threshold = "0.50"
