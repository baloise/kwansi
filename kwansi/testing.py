def test_model(model, test_data, n_tests=5, input_fields=None, output_field=None, metric=None, verbose=False, truncate=50):
    if input_fields is None:
        input_fields = test_data[0].__dict__.keys()
    
    if output_field is None:
        output_field = 'output'  # Default output field name
    
    for idx, example in enumerate(test_data[:n_tests]):
        input_data = {field: getattr(example, field) for field in input_fields}
        prediction = model(**input_data)
        
        if verbose:
            print(f"Test {idx+1}:")
            for field in input_fields:
                print(f"{field.capitalize()}: {getattr(example, field)}")
            print(f"Generated {output_field}: {getattr(prediction, output_field)}")
            
            if metric:
                metric_result = metric(example, prediction)
                print("Metric scores:")
                for key, value in metric_result.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
            
            print("-" * 50)  # Separator for readability
        else:
            output = getattr(prediction, output_field)
            truncated_output = output[:truncate] + "..." if len(output) > truncate else output
            print(f"Test {idx+1}: {truncated_output}")
        
    if not verbose:
        print(f"\nTested {n_tests} examples. Use verbose=True for more details.")