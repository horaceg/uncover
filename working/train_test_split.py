
def indexslice(l, ix):
    for i, item in enumerate(l):
        if i in ix:
            yield item

def mask_ix(l, mask):
    def aux(l, mask):
        for item, bl in zip(l, mask):
            if bl:
                yield item
    return list(aux(l, mask))

def split_train_test(countries_train, countries_test, all_countries):
    mask_train = [c in countries_train for c in all_countries]
    mask_test = [c in countries_test for c in all_countries]
    return mask_train, mask_test

# list(indexslice([1, 2, 3], [0, 2]))
# mask_ix([1, 2, 3], [True, False, False]) ;
