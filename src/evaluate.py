import json


def evaluate(pred_data,gt_data):

    # Initialize a variable to store the number of correct predictions
    correct_predictions = 0

    # Iterate through the ground truths and predicted retrievals
    for gt_item in gt_data['ground_truths']:
        # Find the corresponding prediction for this query id
        for pred_item in pred_data['answers']:
            if gt_item['qid'] == pred_item['qid']:
                # Compare the ground truth 'retrieve' with the predicted 'retrieve'
                if gt_item['retrieve'] == pred_item['retrieve']:
                    correct_predictions += 1
                break

    # Calculate Precision@1
    total_queries = len(gt_data['ground_truths'])
    precision_at_1 = correct_predictions / total_queries

    # Print the result
    print(f'Precision@1: {precision_at_1:.4f}')


if __name__ == "__main__":
    # Load the ground truth data
    with open('../dataset/preliminary/ground_truths_example.json', 'rb') as f:
        gt_data = json.load(f)

    # Load the prediction data
    with open('../output/pred_retrieve.json', 'rb') as f:
        pred_data = json.load(f)


    # Evaluate the model
    evaluate(pred_data, gt_data)