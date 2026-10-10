# ADR-0048: A budget action stops the AWS demo instance when the month's actual spend reaches $20

- **Status:** Accepted
- **Date:** 2026-10-10
- **Decider:** Muhammed Yasin Horasanli
- **Builds on:** [ADR-0042](0042-aws-demo-copy-on-one-ec2-instance.md), [ADR-0047](0047-cloudfront-in-front-of-the-aws-demo-copy.md) (pay-as-you-go accepted without a hard cost cap)

## Context

- The demo copy runs on pay-as-you-go pricing (ADR-0047). The existing budgets (`zero-spend`, `monthly-10`) only send email.
- The account is on AWS's Free plan: $140 of credits on 2026-10-10 ($100 at sign-up plus $40 from two activities), plan end 6 April 2027. AWS can't charge the card on this plan, so until an upgrade a cap protects the credits.
- Frankfurt prices (AWS Pricing API, 2026-10-10): t4g.small $0.0192/h, public IPv4 $0.005/h, gp3 $0.0952/GB-month, CloudFront HTTPS requests $0.012 per 10,000 beyond the free 10 M a month.
- A normal month costs about **$5.55** during the t4g trial (IPv4 + disk) and about **$19.57** always-on after it ($14.02 instance, $3.65 IPv4, $1.90 disk). Expected spend from 10 October to 31 December: at most ~$15.
- Stopping the instance removes the instance and IPv4 charges (an auto-assigned address is released on stop); the disk keeps costing ~$1.90 a month; CloudFront still bills requests, and visitors get an error.
- The decider's requirement: no cut-off before 31 December unless spending is unusually high.

## Options considered

### What happens at $20

#### A: stop the EC2 instance with a budget action (chosen)
- ✅ Native AWS Budgets feature, no code; covers ~90% of a normal month's cost.
- ✅ Free (the first two action-enabled budgets cost nothing).
- ❌ Doesn't stop a CloudFront request flood, which would take ~27 M requests in a month to reach $20.

#### B: A, plus disable the CloudFront distribution (SNS → Lambda)
- ✅ Also caps a request flood.
- ❌ Three more moving parts (SNS topic, Lambda function, IAM) and custom code, for a risk that needs a deliberate attack.

AWS Budgets' newer "spend limits" work through AWS Organizations, which would end the Free plan, so they weren't an option.

### Trigger
- **Actual spend (chosen):** fires only when $20 has really been spent.
- **Forecasted spend:** after the trial a normal month forecasts ~$19.6, so small extras could stop the demo early in a month for nothing.

### Approval
- **Automatic (chosen):** works while nobody is watching.
- **Manual:** an email and a click; useless when the email is missed.

### Where it lives
- **`infra/demo`, beside the instance (chosen):** the action's instance id follows the instance when it is replaced, and the action goes away with `terraform destroy`.
- **A separate stack:** survives a destroy but has to find the instance by tag.

## Decision

`infra/demo/budget.tf`:
- **Budget** `xcrs-demo-stop-at-20`: monthly, all services, credits and refunds excluded; an email at 80%.
- **Budget action:** at **$20 actual**, automatic, runs the SSM "stop EC2 instances" action on the demo instance.
- **Role** `xcrs-demo-budgets-action`:
  - may be assumed only by `budgets.amazonaws.com`, and only for budgets in this account;
  - carries AWS's managed policy `AWSBudgetsActions_RolePolicyForResourceAdministrationWithSSM`, which allows stopping or starting EC2 and RDS instances only through SSM.
- **Email:** the alert address is a variable kept in a gitignored `terraform.tfvars`.

## Trade-offs accepted

- **A request flood isn't capped.** The other budgets still email, and option B stays available.
- **The stop comes hours after the spending,** because budget data refreshes a few times a day.
- **The managed policy reaches every instance in the account** (through SSM only), rather than a hand-written policy limited to one instance. It's maintained by AWS, and the account has one instance.
- **After 31 December a normal always-on month (~$19.57) sits just below the threshold.** Small extras could stop the demo near a month's end. That's acceptable: the demo is disposable, and a restart is one command.

## Revisit when

- **The t4g trial ends (31 December 2026):** choose a nightly stop, destroy-when-idle, or a different threshold.
- **The account is upgraded to a paid plan:** the cap then protects real money; reconsider option B.
- **The demo carries real traffic,** or a flood happens.
