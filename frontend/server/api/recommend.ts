import { handleError } from "vue";
import { RecommendationResult } from "~/models/result";



// const config = useRuntimeConfig()

let environment = process.env.NODE_ENV;
let $endpoint = environment == 'development' ? 'http://localhost:8000' : 'http://159.146.105.19:8000';

console.log('ENVIRONMENT =', process.env.NODE_ENV)
console.log('$ENDPOINTS =', $endpoint)

export default defineEventHandler(async (event) => {
    if (event.method != 'POST') return sendError(event, Error('Unknown parameters'));

    //console.log('event =', event)

    const body = await readBody<{
        took_and_liked: string,
        took_and_neutral: string,
        took_and_disliked: string,
        curious: string
    }>(event);


    //console.log('userData =', body)

    //const {data: responseData} = await useFetch('http://localhost:8000/', {
    const palm_response = await $fetch<RecommendationResult>($endpoint + '/recommendations/palm', {
        method: 'post',
        body: { 
            took_and_liked: body.took_and_liked,
            took_and_neutral: body.took_and_neutral,
            took_and_disliked: body.took_and_disliked,
            curious: body.curious
        }
    })

    const voyage_response = await $fetch<RecommendationResult>($endpoint + '/recommendations/voyage', {
        method: 'post',
        body: { 
            took_and_liked: body.took_and_liked,
            took_and_neutral: body.took_and_neutral,
            took_and_disliked: body.took_and_disliked,
            curious: body.curious
        }
    })

    const mock_response = await $fetch<RecommendationResult>($endpoint + '/recommendations/mock', {
        method: 'post',
        body: { 
            took_and_liked: body.took_and_liked,
            took_and_neutral: body.took_and_neutral,
            took_and_disliked: body.took_and_disliked,
            curious: body.curious
        }
    })


    const save_response = await $fetch<RecommendationResult>($endpoint + '/save_inputs', {
        method: 'post',
        body: { 
            took_and_liked: body.took_and_liked,
            took_and_neutral: body.took_and_neutral,
            took_and_disliked: body.took_and_disliked,
            curious: body.curious
        }
    })

    // console.log(palm_response.status);
    // console.log(palm_response.recommendations[0].role)
    // console.log(palm_response.recommendations[0].explanation)
    // console.log(palm_response.recommendations[0].courses)

    //console.log(voyage_response.recommendations[0].roles[0].role)

    const savedFileName = save_response.fileName
    const readonlyArray = [palm_response.recommendations[0], voyage_response.recommendations[0], mock_response.recommendations[0]]
    //type Element = typeof readonlyArray[number]
    //response.recommendations

    return {
        fileName: savedFileName,
        recommendations: readonlyArray
    } as RecommendationResult;
})
