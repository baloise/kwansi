def test_model(model, test_data, n_tests=5, input_fields=None, output_field=None, verbose=False):
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
            print(f"Character Count: {len(getattr(prediction, output_field))}")
            print("-" * 50)  # Separator for readability
        else:
            print(f"Test {idx+1}: {getattr(prediction, output_field)[:50]}...")  # Show first 50 characters
        
    if not verbose:
        print(f"\nTested {n_tests} examples. Use verbose=True for more details.")