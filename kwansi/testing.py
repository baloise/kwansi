def test_model(model, test_data, n_tests=5, input_fields=None, output_field=None, output_fields=None, evaluator=None, verbose=False, truncate=50):
    if input_fields is None:
        input_fields = test_data[0].__dict__.keys()
    
    if output_fields is None:
        output_fields = [output_field] if output_field else ['output']  # Default output field name
    elif output_field:
        print("Warning: Both output_field and output_fields provided. Using output_fields.")
    
    for idx, example in enumerate(test_data[:n_tests]):
        input_data = {field: getattr(example, field) for field in input_fields}
        prediction = model(**input_data)
        
        if verbose:
            print(f"Test {idx+1}:")
            for field in input_fields:
                print(f"{field.capitalize()}: {getattr(example, field)}")
            
            for field in output_fields:
                print(f"Generated {field}: {getattr(prediction, field)}")
            
            if evaluator:
                evaluator_result = evaluator(example, prediction)
                print("Evaluator scores:")
                for key, value in evaluator_result.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
            
            print("-" * 50)  # Separator for readability
        else:
            for field in output_fields:
                output = getattr(prediction, field)
                truncated_output = output[:truncate] + "..." if len(output) > truncate else output
                print(f"Test {idx+1} - {field}: {truncated_output}")
        
    if not verbose:
        print(f"\nTested {n_tests} examples. Use verbose=True for more details.")