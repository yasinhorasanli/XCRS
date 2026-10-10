# Cost guard (ADR-0048): when the account's actual spend for the month reaches $20 (credits and refunds excluded,
# so usage that credits pay for still counts), AWS Budgets stops the demo instance on its own. A normal month costs
# ~$5.55 during the t4g trial and ~$19.57 always-on after it, so this fires only on unusual spending. Budget data
# refreshes a few times a day: the stop follows the spending by hours. Restart with `aws ec2 start-instances`.

resource "aws_budgets_budget" "demo_stop" {
  name         = "xcrs-demo-stop-at-20"
  budget_type  = "COST"
  limit_amount = tostring(var.stop_at_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_types {
    include_credit = false # credits would hide the spending they pay for
    include_refund = false
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
  }
}

# The identity AWS Budgets uses to run the action. Only the Budgets service, for budgets in this account, may use it.
data "aws_iam_policy_document" "budgets_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["budgets.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = ["arn:aws:budgets::${data.aws_caller_identity.current.account_id}:budget/*"]
    }
  }
}

resource "aws_iam_role" "budgets_action" {
  name               = "xcrs-demo-budgets-action"
  assume_role_policy = data.aws_iam_policy_document.budgets_assume.json
}

# AWS's managed policy for budget actions that stop instances through SSM: it allows stopping and starting EC2 and
# RDS instances only when called via SSM, and only the AWS-Stop/Start documents. The account has one instance.
resource "aws_iam_role_policy_attachment" "budgets_action_ssm" {
  role       = aws_iam_role.budgets_action.name
  policy_arn = "arn:aws:iam::aws:policy/AWSBudgetsActions_RolePolicyForResourceAdministrationWithSSM"
}

resource "aws_budgets_budget_action" "stop_demo_instance" {
  budget_name        = aws_budgets_budget.demo_stop.name
  action_type        = "RUN_SSM_DOCUMENTS"
  approval_model     = "AUTOMATIC"
  notification_type  = "ACTUAL"
  execution_role_arn = aws_iam_role.budgets_action.arn

  action_threshold {
    action_threshold_type  = "ABSOLUTE_VALUE"
    action_threshold_value = var.stop_at_usd
  }

  definition {
    ssm_action_definition {
      action_sub_type = "STOP_EC2_INSTANCES"
      instance_ids    = [aws_instance.demo.id] # follows the instance when it is replaced
      region          = var.region
    }
  }

  subscriber {
    address           = var.alert_email
    subscription_type = "EMAIL"
  }

  depends_on = [aws_iam_role_policy_attachment.budgets_action_ssm]
}
