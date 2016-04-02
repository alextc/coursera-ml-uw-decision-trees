import graphlab


def reached_minimum_node_size(data, min_node_size):
    return len(data) <= min_node_size


def error_reduction(error_before_split, error_after_split):
    return error_before_split - error_after_split


def intermediate_node_num_mistakes(labels_in_node):
    if len(labels_in_node) == 0:
        return 0
    num_of_ones = len(labels_in_node[labels_in_node == 1])
    num_of_minus_ones = len(labels_in_node[labels_in_node == -1])
    if num_of_ones >= num_of_minus_ones:
        return num_of_minus_ones
    else:
        return num_of_ones


def best_splitting_feature(data, features, target):
    best_feature = None
    best_error = 10
    num_data_points = float(len(data))

    for feature in features:
        left_split = data[data[feature] == 0]
        right_split = data[data[feature] == 1]
        left_mistakes = intermediate_node_num_mistakes(left_split[target])
        right_mistakes = intermediate_node_num_mistakes(right_split[target])
        error = (left_mistakes + right_mistakes) / num_data_points
        if error < best_error:
            best_error = error
            best_feature = feature

    return best_feature


def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])


def create_leaf(target_values):
    leaf = {'splitting_feature': None,
            'left': None,
            'right': None,
            'is_leaf': True}

    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])

    if num_ones > num_minus_ones:
        leaf['prediction'] = 1
    else:
        leaf['prediction'] = -1
    return leaf


def decision_tree_create(data, features, target, current_depth=0,
                         max_depth=10, min_node_size=1,
                         min_error_reduction=0.0):
    remaining_features = features[:]  # Make a copy of the features.

    target_values = data[target]
    # print "--------------------------------------------------------------------"
    # print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))

    # Stopping condition 1: All nodes are of the same type.
    if intermediate_node_num_mistakes(target_values) == 0:
        # print "Stopping condition 1 reached. All data points have the same target value."
        return create_leaf(target_values)

    # Stopping condition 2: No more features to split on.
    if remaining_features == []:
        # print "Stopping condition 2 reached. No remaining features."
        return create_leaf(target_values)

    # Early stopping condition 1: Reached max depth limit.
    if current_depth >= max_depth:
        # print "Early stopping condition 1 reached. Reached maximum depth."
        return create_leaf(target_values)

    # Early stopping condition 2: Reached the minimum node size.
    # If the number of data points is less than or equal to the minimum size, return a leaf.
    if reached_minimum_node_size(data, min_node_size):
        # print "Early stopping condition 2 reached. Reached minimum node size."
        return create_leaf(target_values)

    # Find the best splitting feature
    splitting_feature = best_splitting_feature(data, features, target)

    # Split on the best feature that we found.
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]

    # Early stopping condition 3: Minimum error reduction
    # Calculate the error before splitting (number of misclassified examples
    # divided by the total number of examples)
    error_before_split = intermediate_node_num_mistakes(target_values) / float(len(data))

    # Calculate the error after splitting (number of misclassified examples
    # in both groups divided by the total number of examples)
    left_mistakes = intermediate_node_num_mistakes(left_split[target])
    right_mistakes = intermediate_node_num_mistakes(right_split[target])
    error_after_split = (left_mistakes + right_mistakes) / float(len(data))

    # If the error reduction is LESS THAN OR EQUAL TO min_error_reduction, return a leaf.
    if error_reduction(error_before_split, error_after_split) <= min_error_reduction:
        # print "Early stopping condition 3 reached. Minimum error reduction."
        return create_leaf(target_values)

    remaining_features.remove(splitting_feature)
    # print "Split on feature %s. (%s, %s)" % ( \
    #    splitting_feature, len(left_split), len(right_split))

    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target,
                                     current_depth + 1, max_depth, min_node_size, min_error_reduction)

    ## YOUR CODE HERE
    right_tree = decision_tree_create(right_split, remaining_features, target,
                                      current_depth + 1, max_depth, min_node_size, min_error_reduction)

    return {'is_leaf': False,
            'prediction': None,
            'splitting_feature': splitting_feature,
            'left': left_tree,
            'right': right_tree}


def classify(tree, x, annotate = False):
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate:
            print "At leaf, predicting %s" % tree['prediction']
        return tree['prediction']
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate:
            print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)


def evaluate_classification_error(tree, data, label):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x))
    num_of_errors = 0
    for index in range(len(data)):
        if prediction[index] != data[index][label]:
            num_of_errors += 1

    return num_of_errors / float(len(data))


def count_leaves(tree):
    if tree['is_leaf']:
        return 1
    return count_leaves(tree['left']) + count_leaves(tree['right'])


loans = graphlab.SFrame('lending-club-data.gl/')
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x == 0 else -1)
loans = loans.remove_column('bad_loans')

features = ['grade',  # grade of the loan
            'term',  # the term of the loan
            'home_ownership',  # home_ownership status: own, mortgage or rent
            'emp_length',  # number of years of employment
            ]
target = 'safe_loans'
loans = loans[features + [target]]

safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]
percentage = len(risky_loans_raw) / float(len(safe_loans_raw))
safe_loans = safe_loans_raw.sample(percentage, seed=1)
risky_loans = risky_loans_raw
loans_data = risky_loans.append(safe_loans)

for feature in features:
    loans_data_one_hot_encoded = loans_data[feature].apply(lambda x: {x: 1})
    loans_data_unpacked = loans_data_one_hot_encoded.unpack(column_name_prefix=feature)

    # Change None's to 0's
    for column in loans_data_unpacked.column_names():
        loans_data_unpacked[column] = loans_data_unpacked[column].fillna(0)

    loans_data.remove_column(feature)
    loans_data.add_columns(loans_data_unpacked)

features = loans_data.column_names()
features.remove('safe_loans')  # Remove the response variable

train_data, validation_set = loans_data.random_split(.8, seed=1)


my_decision_tree_new = decision_tree_create(train_data, features, 'safe_loans', max_depth=6,
                                            min_node_size=100, min_error_reduction=0.0)

my_decision_tree_old = decision_tree_create(train_data, features, 'safe_loans', max_depth=6,
                                            min_node_size=0, min_error_reduction=-1)

print 'Predicted class via new tree: %s ' % classify(my_decision_tree_new, validation_set[0])
print "Classification path via new tree"
classify(my_decision_tree_new, validation_set[0], annotate = True)
print "Classification path via old tree"
classify(my_decision_tree_old, validation_set[0], annotate = True)

print "Classification error for new tree"
print evaluate_classification_error(my_decision_tree_new, validation_set, 'safe_loans')
print "Classification error for old tree"
print evaluate_classification_error(my_decision_tree_old, validation_set, 'safe_loans')

model_1 = decision_tree_create(train_data, features, 'safe_loans', max_depth=2,
                               min_node_size=0, min_error_reduction=-1)
print "model_1 done"
model_2 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6,
                               min_node_size=0, min_error_reduction=-1)
print "model_2 done"
model_3 = decision_tree_create(train_data, features, 'safe_loans', max_depth=14,
                               min_node_size=0, min_error_reduction=-1)
print "model_3 done"

print "Training data, classification error (model 1):", evaluate_classification_error(model_1, train_data ,'safe_loans')
print "Training data, classification error (model 2):", evaluate_classification_error(model_2, train_data, 'safe_loans')
print "Training data, classification error (model 3):", evaluate_classification_error(model_3, train_data, 'safe_loans')

print "Validation data, classification error (model 1):", evaluate_classification_error(model_1, validation_set ,'safe_loans')
print "Validation data, classification error (model 2):", evaluate_classification_error(model_2, validation_set, 'safe_loans')
print "Validation data, classification error (model 3):", evaluate_classification_error(model_3, validation_set, 'safe_loans')

print "Leaves Count"
print count_leaves(model_1)
print count_leaves(model_2)
print count_leaves(model_3)

model_4 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6,
                               min_node_size=0, min_error_reduction=-1)
print "model_4 done"
model_5 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6,
                               min_node_size=0, min_error_reduction=0)
print "model_5_done"
model_6 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6,
                               min_node_size=0, min_error_reduction=5)
print "model_6_done"

print "Validation data, classification error (model 4):", evaluate_classification_error(model_4, validation_set, 'safe_loans')
print "Validation data, classification error (model 5):", evaluate_classification_error(model_5, validation_set, 'safe_loans')
print "Validation data, classification error (model 6):", evaluate_classification_error(model_6, validation_set, 'safe_loans')

print "Leaves Count"
print count_leaves(model_4)
print count_leaves(model_5)
print count_leaves(model_6)

model_7 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6,
                               min_node_size=0, min_error_reduction=-1)
print "model_7 done"
model_8 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6,
                               min_node_size=2000, min_error_reduction=-1)
print "model_8 done"
model_9 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6,
                               min_node_size=5000, min_error_reduction=-1)
print "model_9 done"

print "Validation data, classification error (model 7):", evaluate_classification_error(model_7, validation_set, 'safe_loans')
print "Validation data, classification error (model 8):", evaluate_classification_error(model_8, validation_set, 'safe_loans')
print "Validation data, classification error (model 9):", evaluate_classification_error(model_9, validation_set, 'safe_loans')

print "Leaves Count"
print count_leaves(model_7)
print count_leaves(model_8)
print count_leaves(model_9)
