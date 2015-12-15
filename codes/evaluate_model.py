#under construction. 
#I do not use this.


file_place = '../data/MSCOCO/annotations/captions_val2014.json'
val_captions,val_caption_id2tokens,val_caption_id2image_id = read_MSCOCO_json(file_place)

#Validiation Set
    print "testing"
    num_val_data=len(val_caption_id2image_id)
    caption_ids_batches=[]
    for caption_length in val_captions.keys():
        caption_ids_set=val_captions[caption_length]
        caption_ids=list(caption_ids_set)
        caption_ids_batches+=[caption_ids[x:x + batchsize] for x in xrange(0, len(caption_ids), batchsize)]
    
    sum_loss = 0
    file_base='../data/MSCOCO/val2014/COCO_val2014_'
    for i, caption_ids_batch in enumerate(caption_ids_batches):
        captions_batch=[val_caption_id2sentence[caption_id] for caption_id in caption_ids_batch]
        sentences=xp.array(captions_batch,dtype=np.int32)
        image_ids_batch=[val_caption_id2image_id[caption_id] for caption_id in caption_ids_batch]

        try:
            images=images_read(image_ids_batch,file_base,volatile=True)
        except Exception as e:
            print 'image reading error'
            print 'type:' + str(type(e))
            print 'args:' + str(e.args)
            print 'message:' + e.message
            print image_ids_batch
            continue

        batchsize=normal_batchsize#becasue I am adusting batch size depending on sentence length, I need to rechange it. 
        if len(caption_ids_batch) != batchsize:
            batchsize=len(caption_ids_batch) 
            #last batch may be less than batchsize. Or depend on caption_length

        loss = forward(images,sentences,volatile=True)
        
        sum_loss      += loss.data * batchsize

    mean_loss     = sum_loss / num_val_data
    print mean_loss
    with open(savedir+"test_mean_loss.txt", "a") as f:
        f.write(str(mean_loss)+'\n')