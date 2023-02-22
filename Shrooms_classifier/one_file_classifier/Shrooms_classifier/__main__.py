
def main():
    import Shrooms_classifier.predict as pred
    prediction = pred.predict_lookup(pred.user_to_predict)
    print(
        f'This mushroom belongs to species "{" ".join(prediction[0][0].split("_"))}" with probability {prediction[0][1]:.3f}.')
    print(
        f'With a lower probability {prediction[1][1]:.3f} this mushroom could also belong to species "{" ".join(prediction[1][0].split("_"))}".')


if __name__ == "__main__":
    main()

